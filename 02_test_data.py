from data1.patch_base import PatchBasedDataset
import glob
import os
from collections import OrderedDict
import paddle
import paddle.nn.functional as F
from simplecv1.util import viz
from paddle.io import DataLoader
from paddle.io import Dataset
from skimage.io import imread, imsave
from simplecv1.api.preprocess import comm
from simplecv1.api.preprocess import segm
from simplecv1.core.config import AttrDict
from paddle.io import SequenceSampler
import numpy as np
from PIL import Image
import paddle
from reprod_log import ReprodLogger
from reprod_log import ReprodDiffHelper



DEFAULT_PATCH_CONFIG = dict(
    patch_size=896,
    stride=512,
)
COLOR_MAP = OrderedDict({
    'background': (0, 0, 0),
    'ship': (0, 0, 63),
    'storage_tank': (0, 191, 127),
    'baseball_diamond': (0, 63, 0),
    'tennis_court': (0, 63, 127),
    'basketball_court': (0, 63, 191),
    'ground_Track_Field': (0, 63, 255),
    'bridge': (0, 127, 63),
    'large_Vehicle': (0, 127, 127),
    'small_Vehicle': (0, 0, 127),
    'helicopter': (0, 0, 191),
    'swimming_pool': (0, 0, 255),
    'roundabout': (0, 63, 63),
    'soccer_ball_field': (0, 127, 191),
    'plane': (0, 127, 255),
    'harbor': (0, 100, 155),
})


class RemoveColorMap(object):
    def __init__(self, color_map=COLOR_MAP, mapping=(1, 2, 3)):
        super(RemoveColorMap, self).__init__()
        self.mapping_mat = np.array(mapping).reshape((3, 1))
        features = np.asarray(list(color_map.values()))
        self.keys = np.matmul(features, self.mapping_mat).squeeze()
        self.labels = np.arange(features.shape[0])

    def __call__(self, image, mask):
        if isinstance(mask, Image.Image):
            mask = np.array(mask, copy=False)

        q = np.matmul(mask, self.mapping_mat).squeeze()

        # loop for each class
        out = np.zeros_like(q)
        for label, k in zip(self.labels, self.keys):
            out += np.where(q == k, label * np.ones_like(q), np.zeros_like(q))

        return image, Image.fromarray(out.astype(np.uint8, copy=False))


class ISAIDSegmmDataset(PatchBasedDataset):
    def __init__(self,
                 image_dir,
                 mask_dir,
                 patch_config=DEFAULT_PATCH_CONFIG,
                 transforms=None):
        super(ISAIDSegmmDataset, self).__init__(image_dir, mask_dir, patch_config, transforms=transforms)

    def generate_path_pair(self):
        image_path_list = glob.glob(os.path.join(self.image_dir, '*.png'))

        mask_path_list = [os.path.join(self.mask_dir, os.path.basename(imfp).replace('.png',
                                                                                     '_instance_color_RGB.png')) for
                          imfp in image_path_list]
        # print("mask--------------------")
        # print(image_path_list,mask_path_list)

        return zip(image_path_list, mask_path_list)

    def show_image_mask(self, idx, mask_on=True, ax=None):
        img_tensor, blob = self[idx]
        img = img_tensor.numpy()
        mask = blob['cls'].numpy()
        if mask_on:
            img = np.where(mask.sum() == 0, img, img * 0.5 + (1 - 0.5) * mask)

        viz.plot_image(img, ax)

    def __getitem__(self, idx):
        img_tensor, y = super(ISAIDSegmmDataset, self).__getitem__(idx)
        # y['cls'] = paddle.cast(y['cls'],dtype='int32')
        return img_tensor, y



class ISAIDSegmmDataLoader(DataLoader):
    def __init__(self, config):
        self.config = AttrDict()
        self.set_defalut()
        self.config.update(config)

        dataset = ISAIDSegmmDataset(self.config.image_dir,
                                    self.config.mask_dir,
                                    self.config.patch_config,
                                    self.config.transforms)

        sampler = SequenceSampler(
            dataset)

        super(ISAIDSegmmDataLoader, self).__init__(dataset,
                                                   self.config.batch_size,
                                                   batch_sampler=sampler,
                                                   num_workers=self.config.num_workers,
                                                  )

    def set_defalut(self):
        self.config.update(dict(
            image_dir='',
            mask_dir='',
            patch_config=dict(
                patch_size=896,
                stride=512,
            ),
            transforms=[
                RemoveColorMap(),
                segm.RandomHorizontalFlip(0.5),
                segm.RandomVerticalFlip(0.5),
                segm.RandomRotate90K((0, 1, 2, 3)),
                segm.FixedPad((896, 896), 255),
                segm.ToTensor(True),
                comm.THMeanStdNormalize((123.675, 116.28, 103.53), (58.395, 57.12, 57.375))
            ],
            batch_size=1,
            num_workers=0,
            training=True
        ))


class ImageFolderDataset(Dataset):
    def __init__(self, image_dir, mask_dir=None):
        self.fp_list = glob.glob(os.path.join(image_dir, '*.png'))
        self.mask_dir = mask_dir
        self.rm_color = RemoveColorMap()

    def __getitem__(self, idx):
        image_np = imread(self.fp_list[idx])
        if self.mask_dir is not None:
            mask_fp = os.path.join(self.mask_dir, os.path.basename(self.fp_list[idx]).replace('.png',
                                                                                              '_instance_color_RGB.png'))
            mask_np = imread(mask_fp)
            _, mask = self.rm_color(None, mask_np)
            mask_np = np.array(mask, copy=False)
        else:
            mask_np = None
        if len(image_np.shape) == 2:
            image_np = np.stack([image_np] * 3, axis=2)
        return image_np, mask_np, os.path.basename(self.fp_list[idx])

    def __len__(self):
        return len(self.fp_list)
import random
def test_forward():
   
    from simplecv1.core.config import AttrDict
    from simplecv1.util.config import import_config
    import paddle
    paddle.disable_static()
    SEED = 2333
    random.seed(SEED)
    np.random.seed(SEED)
    paddle.seed(SEED)

    config_path = 'isaid.factseg'
    cfg = import_config(config_path)
    cfg = AttrDict.from_dict(cfg)
    opts = None
    if opts is not None:
        cfg.update_from_list(opts)
    # train_loader = ISAIDSegmmDataLoader(cfg['data']['train']['params'])
    config = cfg['data']['train']['params']
    # print(config)

    dataset = ISAIDSegmmDataset(config.image_dir,
                                config.mask_dir,
                                config.patch_config,
                                config.transforms)
#     batch_sampler = paddle.io.DistributedBatchSampler(dataset, batch_size=4,  shuffle=False, drop_last=True)
    
    train_loader = paddle.io.DataLoader(
        dataset,
        batch_size=4,
        # batch_sampler=batch_sampler,
        shuffle=False,
        
        num_workers=0,
        return_list=True,
        )
    print(len(train_loader),len(dataset))
    reprod_logger = ReprodLogger()
    import cv2

    for idx, (data, y) in enumerate(train_loader):
        # print(data.shape)
        # img = data[3,:,:,:]
        # img = img.transpose([1,2,0])
        # print(img)

        # cv2.imwrite("1.png",np.array(img).astype(np.uint8))

        # print(img.shape)
        # break
        reprod_logger.add("dataloader_{}_data".format(idx), data.numpy())
        reprod_logger.add("dataloader_{}_mask".format(idx), y['cls'].numpy())

        if idx>0:
           break

    reprod_logger.save("./Step1_5/result/data_paddle.npy")

import numpy as np

if __name__ == '__main__':
        test_forward()

        diff_helper = ReprodDiffHelper()
        torch_info = diff_helper.load_info("./Step1_5/result/data_ref.npy")
        paddle_info = diff_helper.load_info("./Step1_5/result/data_paddle.npy")
        # compare result and produce log
        diff_helper.compare_info(torch_info, paddle_info)
        diff_helper.report(path="./Step1_5/result/log/metric_diff.log")

  
 


