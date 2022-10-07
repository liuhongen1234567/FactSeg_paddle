import argparse
import logging
import paddle
import numpy as np
from data1.isaid import COLOR_MAP
from data1.isaid import ImageFolderDataset
from concurrent.futures import ProcessPoolExecutor
from paddle.io import DataLoader
from simplecv1.api.preprocess import comm
from simplecv1.api.preprocess import segm
from tqdm import tqdm
from simplecv1.data.preprocess import sliding_window
from simplecv1.process.function import  th_divisible_pad
from module.factseg import  FactSeg
from paddle.vision.transforms import functional as F


parser = argparse.ArgumentParser()
parser.add_argument('--config_path', default=None, type=str,
                    help='path to config file')
parser.add_argument('--ckpt_path', default=None, type=str,
                    help='path to model directory')
parser.add_argument('--image_dir', default=None, type=str,
                    help='path to image dir')
parser.add_argument('--mask_dir', default=None, type=str,
                    help='path to mask dir')
parser.add_argument('--vis_dir', default=None, type=str,
                    help='path to vis_dir')
parser.add_argument('--log_dir', default=None, type=str,
                    help='path to log')
parser.add_argument('--patch_size', default=896, type=int,
                    help='patch size')
parser.add_argument('--tta', action='store_true', default=False, help='use tta')

logger = logging.getLogger('SW-Infer')
logger.setLevel(logging.INFO)



class SegmSlidingWinInference(object):
    def __init__(self):
        super(SegmSlidingWinInference, self).__init__()
        self._h = None
        self._w = None

    def patch(self, input_size, patch_size, stride, transforms=None):
        """ divide large image into small patches.

        Returns:

        """
        self.wins = sliding_window(input_size, patch_size, stride)
        self.transforms = transforms
        return self

    def merge(self, out_list):
        pred_list, win_list = list(zip(*out_list))
        num_classes = pred_list[0].shape[1]
        res_img = paddle.zeros([pred_list[0].shape[0], num_classes, self._h, self._w], dtype=paddle.float32)
        res_count = paddle.zeros([self._h, self._w], dtype=paddle.float32)

        for pred, win in zip(pred_list, win_list):
            res_count[win[1]:win[3], win[0]: win[2]] += 1
            res_img[:, :, win[1]:win[3], win[0]: win[2]] += pred.cpu()

        avg_res_img = res_img / res_count

        return avg_res_img

    def forward(self, model, image_np, **kwargs):
        assert self.wins is not None, 'patch must be performed before forward.'
        # set the image height and width
        self._h, self._w, _ = image_np.shape
        return self._forward(model, image_np, **kwargs)

    def _forward(self, model, image_np, **kwargs):
        size_divisor = kwargs.get('size_divisor', None)
        assert self.wins is not None, 'patch must be performed before forward.'
        out_list = []
        for win in self.wins:
                x1, y1, x2, y2 = win
                image_np = np.array(image_np)
                image = image_np[y1:y2, x1:x2, :].astype(np.float32)
                # print(image.shape,"2",image.shape[2]==3,image_np.shape)
                if self.transforms is not None:
                        image = self.transforms(image)
                h, w = image.shape[2:4]
                if size_divisor is not None:
                        # print("before",image.shape,size_divisor)
                        image = th_divisible_pad(image, size_divisor)
                        # print("after", image.shape)

                with paddle.no_grad():
                        if image.shape[1]!=3:
                                print("ERROR")
                        # print(image.shape)
                        out = model(image)
                        # out = paddle.zeros([1,16,image.shape[2],image.shape[3]])
                if size_divisor is not None:
                        out = out[:, :, :h, :w]
                out_list.append((out.cpu(), win))
                paddle.device.cuda.empty_cache()

        self.wins = None

        return self.merge(out_list)



from simplecv1.metric.miou import NPMeanIntersectionOverUnion as NPmIoU
def run():
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

    paddle_state_dict = paddle.load("/home/aistudio/data/data170962/factseg50_paddle.pdparams")
    paddle_model.set_dict(paddle_state_dict)
    paddle_model.eval()

    dataset = ImageFolderDataset(image_dir=args.image_dir, mask_dir=args.mask_dir)

    palette = np.asarray(list(COLOR_MAP.values())).reshape((-1,)).tolist()

    miou_op = NPmIoU(num_classes=16, logdir=None)

    image_trans = comm.Compose([
        segm.ToTensor(True),
        comm.THMeanStdNormalize((123.675, 116.28, 103.53), (58.395, 57.12, 57.375)),
        comm.CustomOp(lambda x: x.unsqueeze(0))
    ])
    segm_helper = SegmSlidingWinInference()

    val_loader = DataLoader(dataset,batch_size=1,shuffle=False,num_workers=0, collate_fn=lambda x:x)

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



if __name__ == '__main__':
    run()
