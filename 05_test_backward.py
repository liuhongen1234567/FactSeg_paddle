import random
import paddle
import numpy as np
from paddle.vision.transforms import functional as F
from reprod_log import ReprodLogger
from reprod_log import ReprodDiffHelper
from module.factseg import  FactSeg
from module.loss import JointLoss
# from paddle.optimizer.lr import PolynomialDecay
from simplecv1.opt.learning_rate import PolynomialDecay1
SEED=2333
paddle.seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


def average_dict(input_dict):
    for k, v in input_dict.items():
        input_dict[k] = v.mean() if v.ndimension() != 0 else v
    return input_dict

def test_backward():

    # load paddle model
    paddle.set_printoptions(precision=20)

    from simplecv1.core.config import AttrDict
    from simplecv1.util.config import import_config
    config_path = 'isaid.factseg'
    cfg = import_config(config_path)
    cfg = AttrDict.from_dict(cfg)
    opts = None
    if opts is not None:
        cfg.update_from_list(opts)
    joint_loss = JointLoss(**cfg['model']['params'].loss.joint_loss)

    paddle_model = FactSeg(cfg['model']['params'])
   
    paddle_state_dict = paddle.load("/home/aistudio/data/data170962/factseg50_paddle.pdparams")
    paddle_model.set_dict(paddle_state_dict)
    paddle_model.eval()

    # init optimizer
    lr_cfg = cfg['learning_rate']['params']
    base_lr = lr_cfg['base_lr']
    power = lr_cfg['power']
    max_iter = 10
    scheduler = PolynomialDecay1(learning_rate=base_lr,decay_steps=10,power=power,cycle=False)

    optimizer_cfg = cfg['optimizer']['params']
    momentum = optimizer_cfg['momentum']
    weight_decay = optimizer_cfg['weight_decay']

#     optimizer = paddle.optimizer.SGD(learning_rate=scheduler, parameters=paddle_model.parameters())

    optimizer = paddle.optimizer.Momentum(learning_rate=scheduler,parameters=paddle_model.parameters(),momentum=momentum,weight_decay=weight_decay,grad_clip= paddle.nn.ClipGradByNorm(clip_norm=cfg['optimizer']['grad_clip']['max_norm']))

    # optimizer = paddle.optimizer.Adam(learning_rate=0.007, beta1=0.9,beta2=0.999,parameters=paddle_model.parameters())

    # prepare logger & load data
    reprod_logger = ReprodLogger()
   
    data_root = '/home/aistudio/Step1_5/data'
    fake_img =paddle.to_tensor(np.load(data_root + '/img.npy'))
    # fake_img = fake_img.squeeze(0)
    for i in range(1):
        fake_img[i] = F.normalize(fake_img[i], mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375))
    # fake_img = F.normalize(fake_img, mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375))
    # fake_img = fake_img.unsqueeze(0)
    cls = paddle.to_tensor(np.load(data_root + '/cls.npy'))
    y ={'cls':cls}


    for idx in range(max_iter):
        # _, loss_paddle = paddle_model(fake_img,y)
        _, loss_paddle = paddle_model(fake_img,y)
        # print(loss_paddle, np.array(scheduler.get_lr()))

        loss_paddle =  average_dict(loss_paddle)
        loss_paddle  = sum([e for e in loss_paddle.values()])
       
        reprod_logger.add("loss_{}".format(idx), loss_paddle.cpu().detach().numpy())
        reprod_logger.add("lr_{}".format(idx),
                          np.array(scheduler.get_lr()))
     

        # reprod_logger.add("loss_{}".format(idx), loss_paddle.cpu().detach().numpy())
        # reprod_logger.add("lr_{}".format(idx),
        #                   np.array([lr]))

        optimizer.clear_grad()
        loss_paddle.backward()
        # print("---------------------------------------------")

        # for name, tensor in paddle_model.named_parameters():
        #     grad = tensor.grad
            # print(name)
            # if grad is not None:
                # print(name, tensor.grad.abs().mean().item())
            # print(name, tensor.grad.abs().mean())
            # break
        optimizer.step()
        # print(features_grad)
        scheduler.step()

    reprod_logger.save("./Step1_5/result/backward_losses_paddle.npy")



# paddle.enable_static()
if __name__ == "__main__":
    test_backward()

    # load data
    
    diff_helper = ReprodDiffHelper()
    torch_info = diff_helper.load_info("./Step1_5/result/backward_losses_ref.npy")
    paddle_info = diff_helper.load_info("./Step1_5/result/backward_losses_paddle.npy")
    
    # compare result and produce log
    diff_helper.compare_info(torch_info, paddle_info)
    diff_helper.report(path="./result/log/backward_diff.log",diff_threshold=1e-4)
