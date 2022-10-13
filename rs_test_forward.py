import paddle
from reprod_log import ReprodDiffHelper
from reprod_log import ReprodLogger
from model_rs_verify.factseg import FactSeg
import numpy as np
from paddle.vision.transforms import functional as F
def test_forward():


    paddle_model =  FactSeg(in_channels=3,num_classes=16,backbone='resnet50',backbone_pretrained=False)
    paddle_model.eval()
    paddle_state_dict = paddle.load("/home/aistudio/data/data171451/factseg50_paddle_RS.pdparams")
    paddle_model.set_dict(paddle_state_dict)

    # load data
    fake_img = np.load("/home/aistudio/Step1_5/data/img.npy")
    fake_img = paddle.to_tensor(fake_img, dtype="float32")
    fake_img = fake_img.squeeze(0)
    fake_img = F.normalize(fake_img, mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375))
    fake_img = fake_img.unsqueeze(0)

    # save the paddle output
    reprod_logger = ReprodLogger()
    paddle_out_rs = paddle_model(fake_img)
    reprod_logger.add("output", paddle_out_rs[0].cpu().detach().numpy())   
    reprod_logger.save("/home/aistudio/Step1_5/result/forward_rs_paddle.npy")


if __name__ == "__main__":

    test_forward()
    
#     load data
    diff_helper = ReprodDiffHelper()
    paddle_rs_info = diff_helper.load_info("/home/aistudio/Step1_5/result/forward_rs_paddle.npy")
    paddle_info = diff_helper.load_info("/home/aistudio/Step1_5/result/forward_paddle.npy")

    # compare result and produce log
    diff_helper.compare_info(paddle_rs_info, paddle_info)
    diff_helper.report(
        path="./Step1_5/result/log/forward_diff.log")
