import torch
import numpy as np
from reprod_log import ReprodLogger
from reprod_log import ReprodDiffHelper
from module.factseg import FactSeg
from module.loss import JointLoss
from torchvision.transforms import functional as F

def test_forward():
    # init loss

    # load paddle model
    from simplecv.core.config import AttrDict
    from simplecv.util.config import import_config
    config_path = 'isaid.factseg'
    cfg = import_config(config_path)
    cfg = AttrDict.from_dict(cfg)
    opts = None
    if opts is not None:
        cfg.update_from_list(opts)
    joint_loss = JointLoss(**cfg['model']['params'].loss.joint_loss)

    torch_model = FactSeg(cfg['model']['params'])
    torch_model.eval()
    ckpt = torch.load("factseg50.pth")
    torch_state_dict = ckpt['model']
    ret = {}
    for k, v in torch_state_dict.items():
        k = k.replace('module.', '')
        ret[k] = v
    torch_model.load_state_dict(ret)



    # prepare logger & load data
    reprod_logger = ReprodLogger()
    data_root = 'torch_ref/fake_data'
    fake_img = torch.tensor(np.load(data_root + '/img.npy'))
    cls = torch.tensor(np.load(data_root + '/cls.npy'))

    for i in range(1):
        fake_img[i] = F.normalize(fake_img[i], mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375))
    y = {'cls': cls}

    _, loss_torch= torch_model(fake_img, y)
    print(loss_torch)
    reprod_logger.add("loss", loss_torch['joint_loss'].cpu().detach().numpy())
    reprod_logger.save("torch_ref/result/loss_ref.npy")

if __name__ == "__main__":
    test_forward()

