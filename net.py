import numpy as np

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L


def add_noise(h, test, sigma=0.2):
    xp = cuda.get_array_module(h.data)
    if test:
        return h
    else:
        return h + sigma * xp.random.randn(*h.data.shape)


class Generator(chainer.Chain):

    def __init__(self, n_hidden, bottom_width=4, ch=512, wscale=0.02):
        self.n_hidden = n_hidden
        self.ch = ch
        self.bottom_width = bottom_width

        w = chainer.initializers.Normal(wscale)

        super(Generator, self).__init__(
            l0=L.Linear(self.n_hidden, bottom_width * bottom_width * ch, initialW=w),
            # //は返り値が整数になる割り算
            dc1=L.Deconvolution2D(ch, ch // 2, ksize=4, stride=2, pad=1, initialW=w),
            dc2=L.Deconvolution2D(ch // 2, ch // 4, ksize=4, stride=2, pad=1, initialW=w),
            dc3=L.Deconvolution2D(ch // 4, ch // 8, ksize=4, stride=2, pad=1, initialW=w),
            dc4=L.Deconvolution2D(ch // 8, 3, ksize=3, stride=1, pad=1, initialW=w),
            bn0=L.BatchNormalization(bottom_width * bottom_width * ch),
            bn1=L.BatchNormalization(ch // 2),
            bn2=L.BatchNormalization(ch // 4),
            bn3=L.BatchNormalization(ch // 8),
        )

    def make_hidden(self, batchsize):
        return np.random.uniform(-1, 1, (batchsize, self.n_hidden, 1, 1)).astype(np.float32)

    def __call__(self, z, test=False):
        # print("z:", z.shape)
        # Deconvolution2Dによって入力ベクトルをCIFAR-10の画像サイズに変換
        h = F.relu(self.bn0(self.l0(z), test=test))
        # print("h0:", h.shape)  # (n, 512*4*4)
        h = F.reshape(h, (z.data.shape[0], self.ch, self.bottom_width, self.bottom_width))
        # print("h1:", h.shape)  # (n, 512, 4, 4)
        h = F.relu(self.bn1(self.dc1(h), test=test))
        # print("h2:", h.shape)  # (n, 256, 8, 8)
        h = F.relu(self.bn2(self.dc2(h), test=test))
        # print("h3:", h.shape)  # (n, 128, 16, 16)
        h = F.relu(self.bn3(self.dc3(h), test=test))
        # print("h4:", h.shape)  # (n, 64, 32, 32)
        x = F.sigmoid(self.dc4(h))
        # print("x:", x.shape)   # (n, 3, 32, 32) = CIFAR-10のサイズ
        return x


class Discriminator(chainer.Chain):

    def __init__(self, bottom_width=4, ch=512, wscale=0.02):
        w = chainer.initializers.Normal(wscale)

        super(Discriminator, self).__init__(
            c0_0=L.Convolution2D(3, ch // 8, 3, 1, 1, initialW=w),
            c0_1=L.Convolution2D(ch // 8, ch // 4, 4, 2, 1, initialW=w),
            c1_0=L.Convolution2D(ch // 4, ch // 4, 3, 1, 1, initialW=w),
            c1_1=L.Convolution2D(ch // 4, ch // 2, 4, 2, 1, initialW=w),
            c2_0=L.Convolution2D(ch // 2, ch // 2, 3, 1, 1, initialW=w),
            c2_1=L.Convolution2D(ch // 2, ch // 1, 4, 2, 1, initialW=w),
            c3_0=L.Convolution2D(ch // 1, ch // 1, 3, 1, 1, initialW=w),
            l4=L.Linear(bottom_width * bottom_width * ch, 1, initialW=w),
            bn0_1=L.BatchNormalization(ch // 4, use_gamma=False),
            bn1_0=L.BatchNormalization(ch // 4, use_gamma=False),
            bn1_1=L.BatchNormalization(ch // 2, use_gamma=False),
            bn2_0=L.BatchNormalization(ch // 2, use_gamma=False),
            bn2_1=L.BatchNormalization(ch // 1, use_gamma=False),
            bn3_0=L.BatchNormalization(ch // 1, use_gamma=False),
        )

    def __call__(self, x, test=False):
        print("x:", x.shape)  # (n, 3, 32, 32)
        h = add_noise(x, test=test)
        print("h0:", h.shape)  # (n, 3, 32, 32)
        h = F.leaky_relu(add_noise(self.c0_0(h), test=test))
        print("h1:", h.shape)  # (n, 64, 32, 32)
        h = F.leaky_relu(add_noise(self.bn0_1(self.c0_1(h), test=test), test=test))
        print("h2:", h.shape)  # (n, 128, 16, 16)
        h = F.leaky_relu(add_noise(self.bn1_0(self.c1_0(h), test=test), test=test))
        print("h3:", h.shape)  # (n, 128, 16, 16)
        h = F.leaky_relu(add_noise(self.bn1_1(self.c1_1(h), test=test), test=test))
        print("h4:", h.shape)  # (n, 256, 8, 8)
        h = F.leaky_relu(add_noise(self.bn2_0(self.c2_0(h), test=test), test=test))
        print("h5:", h.shape)  # (n, 256, 8, 8)
        h = F.leaky_relu(add_noise(self.bn2_1(self.c2_1(h), test=test), test=test))
        print("h6:", h.shape)  # (n, 512, 4, 4)
        h = F.leaky_relu(add_noise(self.bn3_0(self.c3_0(h), test=test), test=test))
        print("h7:", h.shape)  # (n, 512, 4, 4)
        y = self.l4(h)
        print("y:", y.shape)   # (n, 1)

        return y
