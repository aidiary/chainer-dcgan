import chainer
import chainer.functions as F
from chainer import Variable


class DCGANUpdater(chainer.training.StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.gen, self.dis = kwargs.pop('models')
        super(DCGANUpdater, self).__init__(*args, **kwargs)

    def loss_dis(self, dis, y_fake, y_real):
        batchsize = y_fake.data.shape[0]

        # TODO: softmaxではなくsoftplusを使うメリットは？

        # disにとってはy_fake（偽物を入れたときのdisの出力）は小さいほどよい
        # disにとってはy_real（本物を入れたときのdisの出力）は大きいほどよい
        # y_realが大きいほど損失L1は小さくなる
        L1 = F.sum(F.softplus(-y_real)) / batchsize
        L2 = F.sum(F.softplus(y_fake)) / batchsize
        loss = L1 + L2
        chainer.report({'loss': loss}, dis)

        return loss

    def loss_gen(self, gen, y_fake):
        batchsize = y_fake.data.shape[0]

        # genにとってはy_fake (偽物を入れたときのdisの出力)は
        # 大きいほどよい（= 本物だとだませた）
        # y_fakeが大きいほどlossは小さくなる
        loss = F.sum(F.softplus(-y_fake)) / batchsize
        chainer.report({'loss': loss}, gen)

        return loss

    def update_core(self):
        gen_optimizer = self.get_optimizer('gen')
        dis_optimizer = self.get_optimizer('dis')

        batch = self.get_iterator('main').next()
        x_real = Variable(self.converter(batch, self.device)) / 255.
        xp = chainer.cuda.get_array_module(x_real.data)

        gen, dis = self.gen, self.dis
        batchsize = len(batch)

        # 訓練データを入力したときのDの出力
        # この出力は本物のを入れたとき大きくなり、偽物を入れたとき小さくなるとよい
        # 入力が本物である確率ではない（softmaxではなくsoftplusを使っているため）
        y_real = dis(x_real, test=False)

        # Gが生成したデータを入力したときのDの出力
        z = Variable(xp.asarray(gen.make_hidden(batchsize)))
        x_fake = gen(z, test=False)
        y_fake = dis(x_fake, test=False)

        dis_optimizer.update(self.loss_dis, dis, y_fake, y_real)
        gen_optimizer.update(self.loss_gen, gen, y_fake)
