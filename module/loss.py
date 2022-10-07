# Date: 2018.10.28
import paddle.nn as nn
import paddle
import numpy as np
import paddle.nn.functional as F

def cross_entropy_loss(logit, label):
    """
    get cross entropy loss
    Args:
        logit: logit
        label: true label

    Returns:

    """
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logit, label)
    return loss


class InverseWeightCrossEntroyLoss(nn.Layer):
    def __init__(self,
                 class_num,
                 ignore_index=255
                 ):
        super(InverseWeightCrossEntroyLoss, self).__init__()
        self.class_num = class_num
        self.ignore_index=ignore_index

    def forward(self, logit, label):
        """
       get inverse cross entropy loss
        Args:
            logit: a tensor, [batch_size, num_class, image_size, image_size]
            label: a tensor, [batch_size, image_size, image_size]
        Returns:

        """
        inverse_weight = self.get_inverse_weight(label)
        cross_entropy = nn.CrossEntropyLoss(weight=inverse_weight,
                                            ignore_index=self.ignore_index)

        inv_w_loss = cross_entropy(logit, label)
        return inv_w_loss

    def get_inverse_weight(self, label):
        mask = (label >= 0) & (label < self.class_num)
        label = label[mask]
        # reduce dim
        total_num = len(label)
        # get unique label, convert unique label to list
        percentage = paddle.bincount(label, minlength=self.class_num) / float(total_num)
        # get inverse
        w_for_each_class = 1 / paddle.log(1.02 + percentage)
        # convert to tensor
        return w_for_each_class.float()

class FocalLoss(nn.Layer):
    def __init__(self, alpha=None, gamma=1.0, ignore_index=255, reduction=False):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        
        alpha = paddle.to_tensor(self.alpha).reshape([1, -1, 1, 1])


        p = F.softmax(y_pred, axis=1)

        ignore_mask = (y_true == self.ignore_index)

        # one hot encoding
        
        y_index = paddle.clone(y_true)
        y_index[ignore_mask] = 0
        one_hot_y_true = paddle.zeros(y_pred.shape, dtype=paddle.float32)

        one_hot_y_true.scatter_(1, paddle.cast(y_index.unsqueeze(axis=1),dtype='int64'), paddle.ones(one_hot_y_true.shape))

        pt = (p * one_hot_y_true).sum(dim=1)
        modular_factor = (1 - pt).pow(self.gamma)
        
        cls_balance_factor = (alpha.float() * one_hot_y_true.float()).sum(axis=1)
        modular_factor.mul_(cls_balance_factor)

        losses = F.cross_entropy(y_pred, y_true, ignore_index=self.ignore_index, reduction='none')
        losses.mul_(modular_factor)

        if self.reduction:
            valid_mask = (y_true != self.ignore_index).float()
            mean_loss = losses.sum() / valid_mask.sum()
            return mean_loss
        return losses



class DiceLoss(nn.Layer):
    def __init__(self,
                 smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def _dice_coeff(self, pred, target):
        """
        Args:
            pred: [N, 1] within [0, 1]
            target: [N, 1]
        Returns:
        """

        smooth = self.smooth
        inter = paddle.sum(pred * target)
        z = pred.sum() + target.sum() + smooth
        return (2 * inter + smooth) / z

    def forward(self, pred, target):
        return 1. - self._dice_coeff(pred, target)



def som(loss, ratio):
    # 1. keep num
    num_inst = loss.numel()
    num_hns = int(ratio * num_inst)
    # 2. select loss
    top_loss, _ = loss.reshape([-1]).topk(num_hns, -1)
    loss_mask = (top_loss != 0)
    # 3. mean loss

    return paddle.sum(top_loss[loss_mask]) / (loss_mask.sum() + 1e-6)

from paddle.autograd import PyLayer
import numpy as np

class Clone(PyLayer):
    @staticmethod
    def forward(ctx, cls_pred):
        joint_prob = paddle.clone(cls_pred)
        return joint_prob

    @staticmethod
    def backward(ctx, grad_a):
        # print(grad_a.shape,"a")
        x = paddle.zeros_like(grad_a)

        return x

class JointLoss(nn.Layer):
    def __init__(self, ignore_index=255, sample='SOM', ratio=0.2):
        super(JointLoss, self).__init__()
        assert sample in ['SOM', 'OHEM']
        self.ignore_index = ignore_index
        self.sample = sample
        self.ratio = ratio
        print('Sample:', sample)
       

    def forward(self, cls_pred, binary_pred, cls_true, instance_mask=None):
        valid_mask = (cls_true != self.ignore_index)
        fgp = F.sigmoid(binary_pred)
        clsp = F.softmax(cls_pred, axis=1)
        # fgp.register_hook(lambda grad: print('backward fgp', grad.shape, grad.abs().mean()))
        # clsp.register_hook(lambda grad: print('backward clsp', grad.shape, grad.abs().mean()))
        # numerator

        # with paddle.no_grad():
        # joint_prob = paddle.clone(clsp)
        # print(clsp.shape,"clsp")
        joint_prob = Clone.apply(clsp)
        # joint_prob.register_hook(lambda grad: print('backward joint_prob', grad.shape, grad.abs().mean()))
        joint_prob[:, 0, :, :] = (1-fgp).squeeze(axis=1) * clsp[:, 0, :, :]
        joint_prob[:, 1:, :, :] = fgp * clsp[:, 1:, :, :]
        # fgp.register_hook(lambda grad: print('backward fgp11', grad.shape, grad.abs().mean()))
        # clsp.register_hook(lambda grad: print('backward clsp11', grad.shape, grad.abs().mean()))

        # joint_prob.register_hook(lambda grad: print('backward joint_prob1', grad.shape, grad.abs().mean()))
        # # normalization factor, [B x 1 X H X W]
        Z = paddle.sum(joint_prob, axis=1, keepdim=True)
        # Z.register_hook(lambda grad: print('backward z ', grad.shape, grad.abs().mean()))

        # cls prob, [B, N, H, W]
        p_ci = joint_prob / Z
        # print(paddle.log(p_ci).dtype)
        losses = F.nll_loss(paddle.log(p_ci), paddle.cast(cls_true, dtype='int64'), ignore_index=self.ignore_index, reduction='none')
        # losses.register_hook(lambda grad: print('backward losses', grad.shape, grad.abs().mean()))
        
        if self.sample == 'SOM':
            return som(losses, self.ratio)
        elif self.sample == 'OHEM':
            seg_weight = ohem_weight(p_ci, paddle.cast(cls_true, dtype='int64'), thresh=self.ratio)
            return (seg_weight * losses).sum() / seg_weight.sum()
        else:
            return losses.sum() / valid_mask.sum()







def ohem_cross_entropy(y_pred: paddle.Tensor, y_true: paddle.Tensor,
                       ignore_index: int = -1,
                       thresh: float = 0.7,
                       min_kept: int = 100000):
    # y_pred: [N, C, H, W]
    # y_true: [N, H, W]
    # seg_weight: [N, H, W]
    y_true = y_true.unsqueeze(1)
    with paddle.no_grad():
        assert y_pred.shape[2:] == y_true.shape[2:]
        assert y_true.shape[1] == 1
        seg_label = y_true.squeeze(1).long()
        batch_kept = min_kept * seg_label.size(0)
        valid_mask = seg_label != ignore_index
        seg_weight = y_pred.new_zeros(size=seg_label.size())
        valid_seg_weight = seg_weight[valid_mask]

        seg_prob = F.softmax(y_pred, dim=1)

        tmp_seg_label = seg_label.clone().unsqueeze(1)
        tmp_seg_label[tmp_seg_label == ignore_index] = 0
        seg_prob = seg_prob.gather(1, tmp_seg_label).squeeze(1)
        sort_prob, sort_indices = seg_prob[valid_mask].sort()

        if sort_prob.numel() > 0:
            min_threshold = sort_prob[min(batch_kept,
                                          sort_prob.numel() - 1)]
        else:
            min_threshold = 0.0
        threshold = max(min_threshold, thresh)
        valid_seg_weight[seg_prob[valid_mask] < threshold] = 1.

    seg_weight[valid_mask] = valid_seg_weight

    losses = F.cross_entropy(y_pred, y_true.squeeze(1), ignore_index=ignore_index, reduction='none')
    losses = losses * seg_weight

    return losses.sum() / seg_weight.sum()

if __name__ == '__main__':
    paddle.seed(233)
    cls_pred = paddle.randn([2, 5, 4, 4])
    binary_pred = paddle.randn([2, 1, 4, 4])
    cls_true = paddle.ones([2, 4, 4])
    jloss = JointLoss()
    l = jloss(cls_pred, binary_pred, cls_true)
    print(l)
    # y_true = torch.tensor([1, 1, 0]).float()
    # y_pred = torch.tensor([np.nan, 0, 0.2])
    # # l = F.binary_cross_entropy(y_pred, y_true.float(), reduction='none')
    # # l_m = F.cross_entropy(y_pred, y_true.long(), reduction='none')
    # # print(l)
    # # print(l_m)
    #
    # loss = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction='none')
    # print(loss)
    # bceloss = BceWithLogitsLoss()
    # y_pred = torch.tensor([2, 3, 1, 2]).float()
    # y_true = torch.tensor([0, 1, 1, 255]).float()
    # loss = bceloss(torch.clone(y_pred), torch.clone(y_true))
    # print(loss)
    #
    # binary_true = y_true.clone()
    # binary_true[(y_true > 0) * (y_true < 5)] = 1
    # mask = paddle.ones_like(y_true).float()
    # mask[y_true == 255] = 0
    # binary_true[y_true == 255] = 0
    # binary_losses = F.binary_cross_entropy_with_logits(y_pred, binary_true.float(), weight=mask, reduction='none')
    # print(binary_losses.sum() / mask.sum())


    pass