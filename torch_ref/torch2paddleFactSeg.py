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
    torch_path = "factseg50.pth"
    paddle_path = "torch_ref/result/factseg50_paddle.pdparams"
    torch_state_dict = torch.load(torch_path)
    fc_names = ["classifier"]
    paddle_state_dict = {}
    for k in torch_state_dict['model']:
        v = torch_state_dict['model'][k].cpu().numpy()

        k = k.replace('module.',"")
        print(k)

        if False:
            print(k)
        else:
            paddle_state_dict[k] = v
    paddle.save(paddle_state_dict, paddle_path)


if __name__ == "__main__":
    torch2paddle()
