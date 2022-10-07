import paddle
import numpy as np
from reprod_log import ReprodLogger
from reprod_log import ReprodDiffHelper
from module.factseg import  FactSeg
from paddle.vision.transforms import functional as F

from simplecv1.metric.miou import NPMeanIntersectionOverUnion as NPmIoU
# import cv2 
def test_forward():
    # init loss
   

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


    # prepare logger & load data
    reprod_logger = ReprodLogger()
    data_root = '/home/aistudio/Step1_5/data'
    fake_img =paddle.to_tensor(np.load(data_root + '/img.npy'))
#     tmp = fake_img
    fake_img = fake_img.squeeze(0)
    fake_img = F.normalize(fake_img, mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375))
    fake_img = fake_img.unsqueeze(0)


    cls = paddle.to_tensor(np.load(data_root + '/cls.npy'))
    fg_cls =paddle.to_tensor(np.load(data_root + '/fg_cls_label.npy'))
    y ={'cls':cls, 'fg_cls_label':fg_cls}
  
    paddle_out, loss_paddle= paddle_model(fake_img,y)
    paddle_out = paddle_out.argmax(axis=1)
    print(paddle_out.shape, y['cls'].shape)

    miou_op = NPmIoU(num_classes=16, logdir=None)
    miou_op.forward(y['cls'], paddle_out)
    ious, miou = miou_op.summary()
    print(miou.item())
    # print("miou",miou.cpu().detach().numpy(),"iou",ious[1:].cpu().detach().numpy())


    reprod_logger.add("miou", np.array(miou))
    reprod_logger.add("iou", np.array(ious))

    reprod_logger.save("./Step1_5/result/metric_paddle.npy")

if __name__ == "__main__":
    test_forward()

#     load data
    diff_helper = ReprodDiffHelper()
    torch_info = diff_helper.load_info("./Step1_5/result/metric_ref.npy")
    paddle_info = diff_helper.load_info("./Step1_5/result/metric_paddle.npy")
    # compare result and produce log
    diff_helper.compare_info(torch_info, paddle_info)
    diff_helper.report(path="./Step1_5/result/log/metric_diff.log")
