import os.path

import paddle
import random
import numpy as np
import argparse
import logging
from data1.isaid import ISAIDSegmmDataset
from data1.isaid import ImageFolderDataset
from paddle.io import DataLoader
from simplecv1.api.preprocess import comm
from simplecv1.api.preprocess import segm

from module.factseg import  FactSeg
from simplecv1.metric.miou import NPMeanIntersectionOverUnion as NPmIoU
from isaid_eval import SegmSlidingWinInference
import time
from paddleseg.utils import (TimeAverager, calculate_eta, resume, logger,
                             worker_init_fn, train_profiler, op_flops_funs)
from simplecv1.opt.learning_rate import PolynomialDecay1

parser = argparse.ArgumentParser()
parser.add_argument('--config_path', default=None, type=str,
                    help='path to config file')
parser.add_argument('--ckpt_path', default=None, type=str,
                    help='path to model directory')
parser.add_argument('--image_dir', default=None, type=str,
                    help='path to image dir')
parser.add_argument('--mask_dir', default=None, type=str,
                    help='path to mask dir')
parser.add_argument('--resume_model_path', default=None, type=str,
                    help='path to mask dir')
parser.add_argument('--resume_iter', default=0, type=int,
                    help='path to mask dir')
parser.add_argument('--vis_dir', default=None, type=str,
                    help='path to vis_dir')
parser.add_argument('--log_dir', default=None, type=str,
                    help='path to log')
parser.add_argument('--patch_size', default=896, type=int,
                    help='patch size')
parser.add_argument('--tta', action='store_true', default=False, help='use tta')

def average_dict(input_dict):
    list_1 = []
    for k, v in input_dict.items():
        # print(k,v,v.ndimension)
        input_dict[k] = v.mean() if v.ndimension() != 0 else v
        list_1.append(input_dict[k])
    return input_dict, list_1

def evaluate(args,dataset,paddle_model):
    paddle_model.eval()
    segm_helper = SegmSlidingWinInference()


    miou_op = NPmIoU(num_classes=16, logdir=None)

    image_trans = comm.Compose([
        segm.ToTensor(True),
        comm.THMeanStdNormalize((123.675, 116.28, 103.53), (58.395, 57.12, 57.375)),
        comm.CustomOp(lambda x: x.unsqueeze(0))
    ])
    val_loader = DataLoader(dataset,batch_size=1,shuffle=False,num_workers=4, collate_fn=lambda x:x)
    print(len(dataset),"val")
    paddle.device.cuda.empty_cache()
    
    for idx, blob in enumerate(val_loader):
        image, mask, filename = blob[0]

        h, w = image.shape[:2]
        if idx%10==0:
                logging.info('Progress - [{} / {}] size = ({}, {})'.format(idx + 1, len(dataset), h, w))
        seg_helper = segm_helper.patch((h, w), patch_size=(args.patch_size, args.patch_size), stride=512,
                                       transforms=image_trans)
        out = seg_helper.forward(paddle_model, image, size_divisor=32)

        out = out.argmax(axis=1)
        if mask is not None:
            miou_op.forward(mask, out)

    ious, miou = miou_op.summary()

    return miou.item()


def main():
    SEED = 2333
    random.seed(SEED)
    np.random.seed(SEED)
    paddle.seed(SEED)
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler("/home/aistudio/output/train_log.txt")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] : %(message)s ", "%Y-%m-%d %H:%M:%S")
    # logging.filemode='w'
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    args = parser.parse_args()
    # load paddle model
    from simplecv1.core.config import AttrDict
    from simplecv1.util.config import import_config
    config_path = 'isaid.factseg'
    cfg = import_config(config_path)
    cfg = AttrDict.from_dict(cfg)
    opts = None
    if opts is not None:
        cfg.update_from_list(opts)
    paddle_model = FactSeg(cfg['model']['params'])


    lr_cfg = cfg['learning_rate']['params']
    base_lr = lr_cfg['base_lr']
    power = lr_cfg['power']
    scheduler = PolynomialDecay1(learning_rate=base_lr, decay_steps=lr_cfg['max_iters'], power=power, cycle=False)

    optimizer_cfg = cfg['optimizer']['params']
    momentum = optimizer_cfg['momentum']
    weight_decay = optimizer_cfg['weight_decay']


    optimizer = paddle.optimizer.Momentum(learning_rate=scheduler, parameters=paddle_model.parameters(),
                                          momentum=momentum, weight_decay=weight_decay,
                                          grad_clip=paddle.nn.ClipGradByNorm(
                                              clip_norm=cfg['optimizer']['grad_clip']['max_norm']))

    precision ="fp16"
    amp_level ='O1'
    if precision == 'fp16':
        logger.info('use AMP to train. AMP level = {}'.format(amp_level))
        scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
        if amp_level == 'O2':
            model, optimizer = paddle.amp.decorate(
                models=paddle_model,
                optimizers=optimizer,
                level='O2',
                save_dtype='float32')

    start_iter = 0
    batch_size = cfg['data']['train']['params']['batch_size']
    iter1 = start_iter
    num_iters = cfg['train']['num_iters']
    train_config = cfg['data']['train']['params']

    train_dataset = ISAIDSegmmDataset(train_config.image_dir,
                                train_config.mask_dir,
                                train_config.patch_config,
                                train_config.transforms)
 
    print("batch_size",batch_size)
    batch_sampler = paddle.io.DistributedBatchSampler(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    # 后台任务
    train_loader = paddle.io.DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        num_workers=16,
        return_list=True,
        )
    # 线上运行
#     train_loader = paddle.io.DataLoader(
#         train_dataset,
#         batch_sampler=batch_sampler,
#         num_workers=0,
#         return_list=True,
#         )


    # val_dataset = ImageFolderDataset(image_dir=args.image_dir, mask_dir=args.mask_dir)
    print("train_loader",len(train_loader))

    avg_loss = 0.0
    avg_loss_list = []

    iters_per_epoch = len(batch_sampler)
    best_mean_iou = -1.0
    best_model_iter = -1
    reader_cost_averager = TimeAverager()
    batch_cost_averager = TimeAverager()
    batch_start = time.time()
    log_iters = cfg['train']['log_interval_step']
    save_dir="/home/aistudio/output"
    # paddle_model_path='/home/aistudio/data/data170962/fact-seg_temp_15k.pdparams'
    if os.path.exists(args.resume_model_path):
        paddle_state_dict = paddle.load(args.resume_model_path)
        paddle_model.set_dict(paddle_state_dict)
        print("loading from resume model"+args.resume_model_path)
        for i in range(args.resume_iter):
            iter1+=1
            scheduler.step()


    while iter1<num_iters:
        for (img,y) in train_loader:
            paddle_model.train()
            iter1+=1
            if iter1>num_iters:
                break
            reader_cost_averager.record(time.time() - batch_start)
            if precision == 'fp16':
                with paddle.amp.auto_cast(
                        level=amp_level,
                        enable=True,
                        custom_white_list={
                            "elementwise_add", "batch_norm", "sync_batch_norm"
                        },
                        custom_black_list={'bilinear_interp_v2'}):
                    loss = paddle_model(img,y)
                    loss,loss_list = average_dict(loss)
                    loss = sum([e for e in loss.values()])


                scaled = scaler.scale(loss)  # scale the loss
                scaled.backward()  # do backward
                if isinstance(optimizer, paddle.distributed.fleet.Fleet):
                    scaler.minimize(optimizer.user_defined_optimizer, scaled)
                else:
                    scaler.minimize(optimizer, scaled)  # update parameters
            lr = optimizer.get_lr()
            paddle_model.clear_gradients()
            scheduler.step()
            avg_loss += loss.numpy()[0]
        #     print(loss_list,"loss")
            if not avg_loss_list:
                avg_loss_list = [l.numpy() for l in loss_list]
            else:
                for i in range(len(loss_list)):
                    avg_loss_list[i] += loss_list[i].numpy()
            batch_cost_averager.record(
                time.time() - batch_start, num_samples=batch_size)
            if iter1%log_iters==0:
                avg_loss /= log_iters
                avg_loss_list = [l[0] / log_iters for l in avg_loss_list]
                remain_iters = num_iters - iter1
                avg_train_batch_cost = batch_cost_averager.get_average()
                avg_train_reader_cost = reader_cost_averager.get_average()
                eta = calculate_eta(remain_iters, avg_train_batch_cost)
                logger.info(
                    "[TRAIN] epoch: {}, iter: {}/{}, loss: {:.4f}, lr: {:.6f}, batch_cost: {:.4f}, reader_cost: {:.5f}, ips: {:.4f} samples/sec | ETA {}"
                    .format((iter1 - 1
                             ) // iters_per_epoch + 1, iter1, num_iters, avg_loss,
                            lr, avg_train_batch_cost, avg_train_reader_cost,
                            batch_cost_averager.get_ips_average(), eta))
                print("[TRAIN] epoch: {}, iter: {}/{}, loss: {:.4f}, lr: {:.6f}, batch_cost: {:.4f}, reader_cost: {:.5f}, ips: {:.4f} samples/sec | ETA {}"
                    .format((iter1 - 1
                             ) // iters_per_epoch + 1, iter1, num_iters, avg_loss,
                            lr, avg_train_batch_cost, avg_train_reader_cost,
                            batch_cost_averager.get_ips_average(), eta))
                avg_loss = 0.0
                avg_loss_list = []
                reader_cost_averager.reset()
                batch_cost_averager.reset()
        
            if iter1%500==0:
                save_path = os.path.join(save_dir,"fact-seg_temp.pdparams")
                paddle.save(paddle_model.state_dict(),save_path)
            if iter1%5000==0:
                # miou = evaluate(args,val_dataset,paddle_model)
                # msg1 = '[EVAL] epoch: {}, iter: {}/{}, validation mIoU ({:.4f})'.format((iter1 - 1) // iters_per_epoch + 1, iter1,num_iters, best_mean_iou)
                # logger.info(msg1)
                # if miou>best_mean_iou:
                #     best_mean_iou = miou
                #     best_model_iter=iter1

                best_model_path = os.path.join(save_dir,str(iter1)+"_best_model.pdparams")
                print(best_model_path)
                paddle.save(paddle_model.state_dict(), best_model_path)
                
                msg = '[EVAL] The model with the best validation mIoU ({:.4f}) was saved at iter {}.'.format(best_mean_iou, best_model_iter)
                logger.info(msg)
                # paddle.device.cuda.empty_cache()
                print(msg)
            batch_start = time.time()

                # break

if __name__ == '__main__':
    SEED = 2333
    random.seed(SEED)
    np.random.seed(SEED)
    paddle.seed(SEED)
    main()
