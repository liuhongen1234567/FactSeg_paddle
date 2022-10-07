# from paddleseg.transforms import functional as F
from paddle.vision.transforms import functional as F
import paddle
import numpy as np

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target=None):
        if target is None:
            for t in self.transforms:
                image = t(image, target)
            return image

        for t in self.transforms:
          
            image, target = t(image, target)
        #     print(image.shape, target.shape)

        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class CustomOp(object):
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, image, target=None):
        if target is None:
            return self.fn(image)
        return self.fn(image, target)


class THMeanStdNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        mean = np.array(self.mean)[:,np.newaxis,np.newaxis]
        std = np.array(self.std)[:,np.newaxis,np.newaxis]
        image = np.array(image)
        image = F.normalize(image, mean=mean, std=std)

        image = paddle.to_tensor(image,dtype="float32")

        if target is None:
            return image
        return image, target
