from paddle.optimizer.lr import  LRScheduler
import math

class PolynomialDecay1(LRScheduler):

    def __init__(self,
                 learning_rate,
                 decay_steps,
                 end_lr=0.0001,
                 power=1.0,
                 cycle=False,
                 last_epoch=-1,
                 verbose=False):
        assert decay_steps > 0 and isinstance(
            decay_steps, int), " 'decay_steps' must be a positive integer."
        self.decay_steps = decay_steps
        self.end_lr = end_lr
        assert power > 0.0, " 'power' must be greater than 0.0 so that the learning rate will decay."
        self.power = power
        self.cycle = cycle
        super(PolynomialDecay1, self).__init__(learning_rate, last_epoch,
                                              verbose)

    def get_lr(self):
        tmp_epoch_num = self.last_epoch
        tmp_decay_steps = self.decay_steps
        factor = (1 - float(tmp_epoch_num) / float(tmp_decay_steps))**self.power
        cur_lr = self.base_lr*factor
        return cur_lr