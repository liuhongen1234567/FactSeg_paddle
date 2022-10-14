import torch
import paddle
import random
import numpy as np
import re
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
SEED=2019
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
paddle.seed(SEED)

def torch2paddle():
    torch_path = "factseg50.pth"
    paddle_path = "torch_ref/factseg50_paddle_RS.pdparams"
    torch_state_dict = torch.load(torch_path)
    fc_names = ["classifier"]
    paddle_state_dict = {}
    for k in torch_state_dict['model']:
        v = torch_state_dict['model'][k].cpu().numpy()
        print(k)

        k = k.replace('module.',"")
        k = k.replace("running_var", "_variance")
        k = k.replace("running_mean", "_mean")
        
       
        if "fpn" in k:
            k = k.replace("fpn_inner","inner_blocks.")
            k = k.replace("fpn_layer","layer_blocks.")
            digit = re.findall(r'\d',k)
            digit_modify = int(digit[0])-1
            k = k.replace(".{}.".format(int(digit[0])),".{}.".format(digit_modify))
            
#        print(k)

        if False:
            print(k)
        else:
            paddle_state_dict[k] = v
    paddle.save(paddle_state_dict, paddle_path)


if __name__ == "__main__":
    torch2paddle()