import paddle
import paddle.nn as nn
# from simplecv1.interface.configurable import ConfigurableMixin
import re

from simplecv1.util import checkpoint

# _logger = logger.get_logger()

from simplecv1.core.config import AttrDict
class CVModule(nn.Layer):
    __Keys__ = ['GLOBAL', ]

    def __init__(self, config):
        super(CVModule, self).__init__()
        for key in CVModule.__Keys__:
            if key not in config:
                config[key] = dict()

        self._cfg = AttrDict(

        )
        self.set_defalut_config()
        self._cfg.update(config)

    @property
    def config(self):
        return self._cfg

    def forward(self, *input):
        raise NotImplementedError

    def set_defalut_config(self):
        raise NotImplementedError('You should set a default config')

    def init_from_weightfile(self):
        if 'weight' not in self.config.GLOBAL:
            return
        if not isinstance(self.config.GLOBAL.weight, dict):
            return
        if 'path' not in self.config.GLOBAL.weight:
            return
        if self.config.GLOBAL.weight.path is None:
            return

        state_dict = paddle.load(self.config.GLOBAL.weight.path, map_location=lambda storage, loc: storage)
        if checkpoint.is_checkpoint(state_dict):
            state_dict = state_dict[checkpoint.CheckPoint.MODEL]
        ret = {}
        if 'excepts' in self.config.GLOBAL.weight and self.config.GLOBAL.weight.excepts is not None:
            pattern = re.compile(self.config.GLOBAL.weight.excepts)
        else:
            pattern = None

        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k.replace('module.', '')
            if getattr(pattern, 'match', lambda _: False)(k):
                # ignore
                continue
            ret[k] = v

        self.load_state_dict(ret, strict=False)
        print('Load weights from: {}'.format(self.config.GLOBAL.weight.path))
