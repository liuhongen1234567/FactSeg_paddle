import random
import numpy as np

from reprod_log import ReprodLogger
from reprod_log import ReprodDiffHelper
from module.factseg import FactSeg
from module.loss import JointLoss
from simplecv.opt.optimizer import make_optimizer
from simplecv.opt.learning_rate import make_learningrate
import torch
from torchvision.transforms import functional as F
from torch.nn.utils import clip_grad
SEED = 2333
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
from simplecv.module.model_builder import make_model

def average_dict(input_dict):
    for k, v in input_dict.items():
        input_dict[k] = v.mean() if v.ndimension() != 0 else v
    return input_dict


def test_backward():
    # load paddle model
    torch.set_printoptions(precision=20)
    from simplecv.core.config import AttrDict
    from simplecv.util.config import import_config
    config_path = 'isaid.factseg'
    cfg = import_config(config_path)
    cfg = AttrDict.from_dict(cfg)
    opts = None
    if opts is not None:
        cfg.update_from_list(opts)
    joint_loss = JointLoss(**cfg['model']['params'].loss.joint_loss)

#    torch_model = FactSeg(cfg['model']['params'])
    torch_model = make_model(cfg['model'])


    ckpt = torch.load("factseg50.pth")
    torch_state_dict = ckpt['model']
      
    ret = {}
    for k, v in torch_state_dict.items():
        k = k.replace('module.', '')
        ret[k] = v
    torch_model.load_state_dict(ret)
    torch_model.eval()

    # init optimizer
    max_iter = 10
    scheduler= make_learningrate(cfg['learning_rate'])
    cfg['optimizer']['params']['lr'] = scheduler.base_lr
    optimizer = make_optimizer(cfg['optimizer'], params=torch_model.parameters())
    
#    optimizer = torch.optim.Adam(lr=0.007, params=torch_model.parameters())


    # prepare logger & load data
    reprod_logger = ReprodLogger()

    data_root = 'torch_ref/fake_data'
    fake_img = torch.tensor(np.load(data_root + '/img.npy'))

    for i in range(1):
        fake_img[i] = F.normalize(fake_img[i], mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375))
    cls = torch.tensor(np.load(data_root + '/cls.npy'))
    y = {'cls': cls}
    _global_step = 0

    for idx in range(max_iter):

        torch.nn.utils.clip_grad_norm_(torch_model.parameters(), max_norm=35, norm_type=2)

        _, loss_torch = torch_model(fake_img, y)
       

        loss_torch = average_dict(loss_torch)
        loss_torch = sum([e for e in loss_torch.values()])
        for param_group in optimizer.param_groups:
            temp_lr = param_group['lr']
        print(loss_torch, temp_lr )
        _global_step+=1
        reprod_logger.add("loss_{}".format(idx), loss_torch.cpu().detach().numpy())
        reprod_logger.add("lr_{}".format(idx),
                          np.array( temp_lr))

        optimizer.zero_grad()

        loss_torch.backward()
        for name, tensor in  torch_model.named_parameters():
            grad = tensor.grad
        optimizer.step()
        scheduler.step(_global_step,optimizer )

    reprod_logger.save("torch_ref/result/backward_losses_ref.npy")


# paddle.enable_static()
if __name__ == "__main__":
    test_backward()
