import paddle
import paddle.nn as nn

GlobalAvgPool2D = lambda: nn.AdaptiveAvgPool2D(1)

class GlobalAvgPool2DBaseline(nn.Layer):
    def __init__(self):
        super(GlobalAvgPool2DBaseline, self).__init__()

    def forward(self, x):
        x_pool = paddle.mean(x.view(x.size(0), x.size(1), x.size(2) * x.size(3)), axis=2)

        x_pool = x_pool.view(x.size(0), x.size(1), 1, 1).contiguous()
        return x_pool
