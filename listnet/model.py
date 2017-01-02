from __future__ import absolute_import, division, print_function

import chainer
import chainer.functions as F
import chainer.links as L
import chainer.initializers as I
import numpy as np


class ListNet(chainer.Chain):
    def __init__(self, input_size, n_units, dropout):
        super(ListNet, self).__init__(
            l1=L.Linear(input_size, n_units, initialW=I.GlorotNormal()),
            l2=L.Linear(n_units, 1, initialW=I.GlorotNormal(),
                        nobias=True))
        self.add_persistent("_dropout", dropout)

    def __call__(self, x, train=True):
        s = list(x.shape)
        n_tokens = np.prod(s[:-1])
        x = F.reshape(x, (n_tokens, -1))
        if self._dropout > 0.:
            x = F.dropout(x, self._dropout, train=train)
        o_1 = F.tanh(self.l1(x))

        if self._dropout > 0.:
            o_1 = F.dropout(o_1, self._dropout, train=train)
        o_2 = self.l2(o_1)

        return F.reshape(o_2, s[:-1])
