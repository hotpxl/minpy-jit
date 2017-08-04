import mxnet.ndarray as nd
import sys
import minpy.nn as nn


class Sigmoid(nn.Module):
    def forward(self, x):
        return .5 * (self.F.tanh(.5 * x) + 1)


class LSTM(nn.Module):
    def __init__(self, in_features, h_features, out_features):
        super().__init__()
        XSHAPE = (in_features, h_features)
        HSHAPE = (h_features, h_features)
        self._sigmoid = Sigmoid()
        self._xi, self._hi = nn.Linear(*XSHAPE), nn.Linear(*HSHAPE)
        self._xf, self._hf = nn.Linear(*XSHAPE), nn.Linear(*HSHAPE)
        self._xo, self._ho = nn.Linear(*XSHAPE), nn.Linear(*HSHAPE)
        self._xg, self._hg = nn.Linear(*XSHAPE), nn.Linear(*HSHAPE)

    def forward(self, x, h, c):
        i = self._sigmoid(self._xi(x) + self._hi(h))
        f = self._sigmoid(self._xf(x) + self._hf(h))
        o = self._sigmoid(self._xo(x) + self._ho(h))
        g = self._sigmoid(self._xg(x) + self._hg(h))
        c = f * c + i * g
        h = o * self.F.tanh(c)
        return h, c


model = LSTM(28 * 28, 64, 10)
model.compile()
data = nd.zeros((10, 28 * 28))
h = nd.zeros((10, 64))
c = nd.zeros((10, 64))
h, c = model(data, h, c)
