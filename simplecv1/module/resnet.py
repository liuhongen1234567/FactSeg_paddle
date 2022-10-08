import paddle
import paddle.nn as nn
# from torch.utils import checkpoint as cp
from functools import partial
# from paddle.vision.models import  resnet50
from simplecv1.module._resnet import resnet50
from simplecv1.interface import CVModule
from  simplecv1.util import param_util
# import param_util
from simplecv1.module import context_block

def make_layer(block, in_channel, basic_out_channel, blocks, stride=1, dilation=1):
    downsample = None
    if stride != 1 or in_channel != basic_out_channel * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2D(in_channel, basic_out_channel * block.expansion,
                      kernel_size=1, stride=stride),
            nn.BatchNorm2D(basic_out_channel * block.expansion),
        )

    layers = []
    layers.append(block(in_channel, basic_out_channel, stride, dilation, downsample))
    in_channel = basic_out_channel * block.expansion
    for i in range(1, blocks):
        layers.append(block(in_channel, basic_out_channel, dilation=dilation))

    return nn.Sequential(*layers)
# nn.Layer
class ResNetEncoder(CVModule.CVModule):

    def __init__(self,
                 config):
        super(ResNetEncoder, self).__init__(config)
        if all([self.config.output_stride != 16,
                self.config.output_stride != 32,
                self.config.output_stride != 8]):
            raise ValueError('output_stride must be 8, 16 or 32.')

        self.resnet = resnet50(pretrained=self.config.pretrained,norm_layer=self.config.norm_layer)
        # self.resnet = resnet50(pretrained=False, norm_layer=self.config.norm_layer)
        # self.resnet = resnet50(pretrained=False)
        # self.resnet._children

        # self.resnet = registry.MODEL[self.config.resnet_type](pretrained=self.config.pretrained,
        #                                                       norm_layer=self.config.norm_layer)
        print('ResNetEncoder: pretrained = {}'.format(self.config.pretrained))
        # self.resnet.children().pop('fc')
        self.resnet._sub_layers.pop('fc')
        if not self.config.batchnorm_trainable:
            self._frozen_res_bn()

        self._freeze_at(at=self.config.freeze_at)

        if self.config.output_stride == 16:
            self.resnet.layer4.apply(partial(self._nostride_dilate, dilate=2))
        elif self.config.output_stride == 8:
            self.resnet.layer3.apply(partial(self._nostride_dilate, dilate=2))
            self.resnet.layer4.apply(partial(self._nostride_dilate, dilate=4))

    @property
    def layer1(self):
        return self.resnet.layer1

    @layer1.setter
    def layer1(self, value):
        del self.resnet.layer1
        self.resnet.layer1 = value

    @property
    def layer2(self):
        return self.resnet.layer2

    @layer2.setter
    def layer2(self, value):
        del self.resnet.layer2
        self.resnet.layer2 = value

    @property
    def layer3(self):
        return self.resnet.layer3

    @layer3.setter
    def layer3(self, value):
        del self.resnet.layer3
        self.resnet.layer3 = value

    @property
    def layer4(self):
        return self.resnet.layer4

    @layer4.setter
    def layer4(self, value):
        del self.resnet.layer4
        self.resnet.layer4 = value

    def _frozen_res_bn(self):
        print('ResNetEncoder: freeze all BN layers')
        param_util.freeze_modules(self.resnet, nn.BatchNorm2D)
        for m in self.resnet.modules():
            if isinstance(m, nn.BatchNorm2D):
                m.eval()

    def _freeze_at(self, at=2):
        if at >= 1:
            param_util.freeze_params(self.resnet.conv1)
            param_util.freeze_params(self.resnet.bn1)
        if at >= 2:
            param_util.freeze_params(self.resnet.layer1)
        if at >= 3:
            param_util.freeze_params(self.resnet.layer2)
        if at >= 4:
            param_util.freeze_params(self.resnet.layer3)
        if at >= 5:
            param_util.freeze_params(self.resnet.layer4)

    @staticmethod
    def get_function(module):
        def _function(x):
            y = module(x)
            return y

        return _function

    def forward(self, inputs):
        x = inputs
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        # os 4, #layers/outdim: 18,34/64; 50,101,152/256
        if self.config.with_cp[0] and not x.stop_gradient:
            pass
        else:
            c2 = self.resnet.layer1(x)
        # os 8, #layers/outdim: 18,34/128; 50,101,152/512
        if self.config.with_cp[1] and not c2.stop_gradient:
            pass
        else:
            c3 = self.resnet.layer2(c2)
        # os 16, #layers/outdim: 18,34/256; 50,101,152/1024
        if self.config.with_cp[2] and not c3.stop_gradient:
            pass
        else:
            c4 = self.resnet.layer3(c3)
        # os 32, #layers/outdim: 18,34/512; 50,101,152/2048
        if self.config.include_conv5:
            c5 = self.resnet.layer4(c4)
            return [c2, c3, c4, c5]

        return [c2, c3, c4]

    def set_defalut_config(self):
        self.config.update(dict(
            resnet_type='resnet50',
            include_conv5=True,
            batchnorm_trainable=True,
            pretrained=False,
            freeze_at=0,
            # 16 or 32
            output_stride=32,
            with_cp=(False, False, False, False),
            norm_layer=nn.BatchNorm2D,
        ))

    def train(self, mode=True):
        super(ResNetEncoder, self).train(mode)
        self._freeze_at(self.config.freeze_at)
        if mode and not self.config.batchnorm_trainable:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, nn.LayerList.batchnorm._BatchNorm):
                    m.eval()

    def _nostride_dilate(self, m, dilate):
        # ref:
        # https://github.com/CSAILVision/semantic-segmentation-pytorch/blob/1235deb1d68a8f3ef87d639b95b2b8e3607eea4c/models/models.py#L256
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)


def plugin_context_block2d(module: nn.Layer, ratio):
    """

    Args:
        module: (nn.Layer): containing module
        ratio: (float) reduction ratio

    Returns:
        The original module with the converted `context_block.Bottleneck` layer

    Example::

    """
    classname = module.__class__.__name__
    module_output = module
    if classname.find('Bottleneck') != -1:
        module_output = context_block.Bottleneck(module.conv1.in_channels,
                                                 module.conv1.out_channels,
                                                 ratio=ratio,
                                                 stride=module.stride,
                                                 downsample=module.downsample)
        # conv1 bn1
        param_util.copy_conv_parameters(module.conv1, module_output.conv1)
        if isinstance(module.bn1, nn.BatchNorm2D):
            param_util.copy_bn_parameters(module.bn1, module_output.bn1)
        elif isinstance(module.bn1, nn.GroupNorm):
            param_util.copy_weight_bias_attr(module.bn1, module_output.bn1)
        # conv2 bn2
        param_util.copy_conv_parameters(module.conv2, module_output.conv2)
        if isinstance(module.bn2, nn.BatchNorm2D):
            param_util.copy_bn_parameters(module.bn2, module_output.bn2)
        elif isinstance(module.bn2, nn.GroupNorm):
            param_util.copy_weight_bias_attr(module.bn2, module_output.bn2)
        # conv3 bn3
        param_util.copy_conv_parameters(module.conv3, module_output.conv3)
        if isinstance(module.bn3, nn.BatchNorm2D):
            param_util.copy_bn_parameters(module.bn3, module_output.bn3)
        elif isinstance(module.bn3, nn.GroupNorm):
            param_util.copy_weight_bias_attr(module.bn3, module_output.bn3)
        del module
        return module_output

    for name, sub_module in module.named_children():
        module_output.add_module(name, plugin_context_block2d(sub_module, ratio))
    del module
    return module_output


from simplecv1.core.config import AttrDict
from simplecv1.util.config import import_config
def main():
    x = paddle.rand([2,3,512,512])
    config_path='isaid.farseg50'
    cfg = import_config(config_path)
    cfg = AttrDict.from_dict(cfg)
    opts = None
    if opts is not None:
        cfg.update_from_list(opts)
    res = ResNetEncoder(cfg['model']['params'].resnet_encoder)
    out = res(x)
    for o in out:
        print(o.shape,"o")

    # x
    print(cfg['model']['params'].resnet_encoder.with_cp)
    print("Yes")
if __name__=='__main__':
    main()
