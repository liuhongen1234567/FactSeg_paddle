import paddle
import numpy as np
from reprod_log import ReprodLogger
from reprod_log import ReprodDiffHelper
from module.factseg import  FactSeg
from paddle.vision.transforms import functional as F
import cv2 
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
#     data_root ='/home/aistudio/Step1_5/0_test_data'
    data_root = '/home/aistudio/Step1_5/data'
    fake_img =paddle.to_tensor(np.load(data_root + '/img.npy'))
#     tmp = fake_img
    fake_img = fake_img.squeeze(0)
    fake_img = F.normalize(fake_img, mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375))
    fake_img = fake_img.unsqueeze(0)


    cls = paddle.to_tensor(np.load(data_root + '/cls.npy'))
    y ={'cls':cls}
  
    _, loss_paddle= paddle_model(fake_img,y)

    reprod_logger.add("loss", loss_paddle['joint_loss'].cpu().detach().numpy())
    reprod_logger.save("./Step1_5/result/loss_paddle.npy")

#     fake_img = tmp.transpose([0,2,3,1])
#     fake_img = np.array(fake_img)[0]
#     cv2.imwrite('2.png',np.array(fake_img,dtype='uint8'))

#     pred = paddle.argmax(loss_paddle,axis=1)
#     pred = paddle.squeeze(pred)
#     pred = pred.numpy().astype('uint8')
#     print("loss",pred.shape,cls.shape)
#     save_img = loss_paddle
#     img = save_img[0,1,:,:]
#     img = np.array(img)

#     from paddleseg.utils import logger, progbar, visualize
#     from paddleseg import utils
#     color_map = visualize.get_color_map_list(256)
#     im_path='2.png'
#     added_image =  utils.visualize.visualize(
#             im_path, pred, color_map, weight=0.6)
#     added_image_path = 'vis.png'
#     # mkdir(added_image_path)
#     cv2.imwrite(added_image_path, added_image)

if __name__ == "__main__":
    test_forward()

#     load data
    diff_helper = ReprodDiffHelper()
    torch_info = diff_helper.load_info("./Step1_5/result/loss_paddle.npy")
    paddle_info = diff_helper.load_info("./Step1_5/result/loss_ref.npy")
    # compare result and produce log
    diff_helper.compare_info(torch_info, paddle_info)
    diff_helper.report(path="./Step1_5/result/log/loss_diff.log")
