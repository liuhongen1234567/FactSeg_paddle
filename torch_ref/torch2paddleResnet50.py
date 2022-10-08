import torch
import paddle
import random
import numpy as np

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
SEED=2019
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
paddle.seed(SEED)

def torch2paddle():
    torch_path = "resnet50-19c8e357.pth"
    paddle_path = "torch_ref/result/resnet50_paddle.pdparams"
    torch_state_dict = torch.load(torch_path,map_location=torch.device('cpu'))
    fc_names = ["fc"]
    paddle_state_dict = {}
    for k in torch_state_dict:

        if "num_batches_tracked" in k:
            continue
        v = torch_state_dict[k].detach().cpu().numpy()
        flag = [i in k for i in fc_names]
        # classfier 第二个是Relu要排除
        if any(flag) and "weight" in k and (not ".1" in k):  # ignore bias

            new_shape = [1, 0] + list(range(2, v.ndim))
            v = v.transpose(new_shape)
        k = k.replace("running_var", "_variance")
        k = k.replace("running_mean", "_mean")
        print(k)

        if False:
            print(k)
        else:
            paddle_state_dict[k] = v
    paddle.save(paddle_state_dict, paddle_path)


if __name__ == "__main__":
    torch2paddle()
