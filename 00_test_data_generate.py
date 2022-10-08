import argparse
import logging
import paddle
import numpy as np
from data1.isaid import ISAIDSegmmDataset

def run():
    # load paddle model
    from simplecv1.core.config import AttrDict
    from simplecv1.util.config import import_config
    config_path = 'isaid.factseg'
    cfg = import_config(config_path)
    cfg = AttrDict.from_dict(cfg)
    opts = None
    if opts is not None:
        cfg.update_from_list(opts)
    train_config = cfg['data']['train']['params']
    train_dataset = ISAIDSegmmDataset(train_config.image_dir,
                                train_config.mask_dir,
                                train_config.patch_config,
                                train_config.transforms)
    # 后台任务
    train_loader = paddle.io.DataLoader(
        train_dataset,
        batch_size = 1,
        num_workers=0,
        return_list=True,
        )
    print(len(train_loader))
    save_path ='/home/aistudio/Step1_5/0_test_data/'
    data, mask = next(iter(train_loader))
    # for (data, mask) in train_loader:
    print(data.shape, mask['cls'].shape)
    np.save(save_path+'img.npy', np.array(data))
    np.save(save_path+'cls.npy',np.array(mask['cls']))


if __name__ == '__main__':
    run()
