import paddle
from reprod_log import ReprodDiffHelper
from reprod_log import ReprodLogger
from module.factseg import FactSeg
import numpy as np
import torch
from torchvision.transforms import functional as F
def test_forward():
    # load paddle model
    from simplecv.core.config import AttrDict
    from simplecv.util.config import import_config
    config_path = 'isaid.factseg'
    cfg = import_config(config_path)
    cfg = AttrDict.from_dict(cfg)
    opts = None
    if opts is not None:
        cfg.update_from_list(opts)

    torch_model = FactSeg(cfg['model']['params'])
    ckpt = torch.load("factseg50.pth")
    torch_state_dict = ckpt['model']
    ret = {}
    for k, v in torch_state_dict.items():
        k = k.replace('module.', '')
        ret[k] = v
    torch_model.load_state_dict(ret)
    torch_model.eval()
  
    inputs = np.load("torch_ref/fake_data/img.npy")
    inputs =torch.tensor(inputs, dtype=torch.float32)
    for i in range(1):
        inputs[i] = F.normalize(inputs[i], mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375))

    # save the paddle output
    reprod_logger = ReprodLogger()
    print(inputs.shape)
    torch_out = torch_model(inputs)
    reprod_logger.add("output", torch_out.cpu().detach().numpy())
    reprod_logger.save("torch_ref/result/forward_ref.npy")


if __name__ == "__main__":
    test_forward()


    # # compare result and produce log
    # diff_helper.compare_info(torch_info, paddle_info)
    # diff_helper.report(
    #     path="./Step1_5/result/log/forward_diff.log", diff_threshold=1e-5)
