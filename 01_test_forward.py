import paddle
from reprod_log import ReprodDiffHelper
from reprod_log import ReprodLogger
from module.factseg import  FactSeg
import numpy as np
from paddle.vision.transforms import functional as F
def test_forward():

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
    paddle_model.eval()
    paddle_state_dict = paddle.load("/home/aistudio/data/data170962/factseg50_paddle.pdparams")
    paddle_model.set_dict(paddle_state_dict)

    # load data
    fake_img = np.load("/home/aistudio/Step1_5/data/img.npy")
    fake_img = paddle.to_tensor(fake_img, dtype="float32")
    fake_img = fake_img.squeeze(0)
    fake_img = F.normalize(fake_img, mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375))
    fake_img = fake_img.unsqueeze(0)

    # save the paddle output
    reprod_logger = ReprodLogger()
    paddle_out = paddle_model(fake_img)
    reprod_logger.add("output", paddle_out.cpu().detach().numpy())
    reprod_logger.save("./Step1_5/result/forward_paddle.npy")


if __name__ == "__main__":

    test_forward()
    
#     load data
    diff_helper = ReprodDiffHelper()
    torch_info = diff_helper.load_info("./Step1_5/result/forward_ref.npy")
    paddle_info = diff_helper.load_info("./Step1_5/result/forward_paddle.npy")

    # compare result and produce log
    diff_helper.compare_info(torch_info, paddle_info)
    diff_helper.report(
        path="./Step1_5/result/log/forward_diff.log")
