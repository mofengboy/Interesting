import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from GAN.Mnist.DataSet import getMnistTrain


class Discriminator:
    def __init__(self, data_set, device, batch_size, lr, model_path):
        self.data_set = data_set
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.lr = lr
        self.model_path = model_path
        self.model = self.DModel()
        self.model.to(self.device)
        self.loss_func = torch.nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    class DModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 32, (3, 3), (1, 1))
            self.conv2 = nn.Conv2d(32, 64, (3, 3), (1, 1))
            self.mp = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
            self.fc1 = nn.Linear(1600, 1024)
            self.fc2 = nn.Linear(1024, 128)
            self.fc3 = nn.Linear(128, 1)
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = x.reshape(-1, 1, 28, 28)
            y = self.conv1(x)
            y = self.relu(y)
            y = self.mp(y)
            y = self.conv2(y)
            y = self.relu(y)
            y = self.mp(y)

            y = y.reshape(-1, 1600)
            y = self.relu(self.fc1(y))
            y = self.relu(self.fc2(y))
            y = self.fc3(y)
            y = self.sigmoid(y)
            return y

    def train(self, data=torch.zeros(1)):
        self.model.train()
        data = data.to(self.device)
        # 读取数据
        if torch.zeros(1).to(self.device).equal(data):
            trainData, target = getMnistTrain(self.data_set, self.batch_size)
            trainData = trainData.to(self.device).float()
            target = torch.ones([self.batch_size, 1]).to(self.device)
        else:
            trainData = data.to(self.device).float()
            target = torch.zeros([self.batch_size, 1]).to(self.device)

        out = self.model(trainData)
        # print(out)

        loss = self.loss_func(out, target)

        # 梯度清零
        self.optimizer.zero_grad()

        loss.backward()
        # 更新参数
        self.optimizer.step()
        return loss


class Generator:
    def __init__(self, device, batch_size, lr, model_path="", load_model_path=""):
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.lr = lr
        self.model_path = model_path
        if load_model_path != "":
            self.model = torch.load(load_model_path)
        else:
            self.model = self.GModel(batch_size, device)
        self.model.to(self.device)
        self.loss_func = torch.nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    class GModel(nn.Module):
        def __init__(self, batch_size, device):
            super().__init__()
            self.batch_size = batch_size
            self.device = torch.device(device)

            self.fc1 = nn.Linear(64, 128)
            self.fc2 = nn.Linear(128, 128 * 3 * 3)
            self.fc3 = nn.Linear(128 * 3 * 3, 128 * 7 * 7)
            self.relu = nn.ReLU()
            self.bn1 = nn.BatchNorm1d(128)
            self.bn2 = nn.BatchNorm1d(128 * 3 * 3)
            self.bn3 = nn.BatchNorm1d(128 * 7 * 7)
            self.bn4 = nn.BatchNorm2d(64)

            self.upSample1 = nn.Upsample(scale_factor=4)
            self.upSample2 = nn.Upsample(scale_factor=2)
            self.conv1 = nn.Conv2d(128, 64, (3, 3), (1, 1), (2, 2))
            self.conv2 = nn.Conv2d(64, 1, (5, 5), (2, 2))
            self.sigmoid = nn.Sigmoid()

        def forward(self, bz=0):
            if bz == 0:
                bz = self.batch_size
            x = torch.rand([bz, 64]).to(self.device).float() * 100
            x = self.fc1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.fc2(x)
            x = self.bn2(x)
            x = self.relu(x)
            x = self.fc3(x)
            x = self.bn3(x)
            x = self.relu(x)

            x = x.reshape(-1, 128, 7, 7)
            x = self.upSample1(x)
            x = self.conv1(x)
            x = self.bn4(x)
            x = self.relu(x)
            x = self.upSample2(x)
            x = self.conv2(x)
            x = self.sigmoid(x)
            return x

    def train(self, D):
        targets = torch.ones([self.batch_size, 1]).to(self.device)
        # 定义交叉熵损失
        loss_func = torch.nn.BCELoss()

        g_output = self.model.forward()

        # 输入鉴别器
        d_output = D.model.forward(g_output)

        loss = loss_func(d_output, targets)
        # 梯度清零
        self.optimizer.zero_grad()

        loss.backward()
        # 更新参数
        self.optimizer.step()

        return loss


def showImg(data):
    data = data.numpy() * 255
    data = data.reshape(-1, 28)
    plt.imshow(data)
    plt.show()


def test():
    pass


def train():
    dataSet = "../data/mnist_all.mat"
    DBatchSize = 16
    GBatchSize = 256
    lr = 0.001
    device = "cuda"
    GModelPath = "GModel"
    DModelPath = "DModel"
    D = Discriminator(data_set=dataSet, device=device, batch_size=DBatchSize, lr=lr, model_path=DModelPath)
    G = Generator(device=device, batch_size=GBatchSize, lr=lr, model_path=GModelPath)

    writer = SummaryWriter('./log')

    for i in range(3000):
        lossTrue = D.train()
        lossFalse = D.train(data=G.model.forward(DBatchSize).detach())
        lossG = G.train(D)
        lossG = G.train(D)
        if i % 200 == 0:
            print("当前为第" + str(i) + "次")
            print("鉴别器真损失值为：")
            print(lossTrue)
            writer.add_scalar("lossTrue", lossTrue, i)
            print("鉴别器假损失值为：")
            print(lossFalse)
            writer.add_scalar("lossFalse", lossFalse, i)
            print("生成器损失值为：")
            print(lossG)
            writer.add_scalar("lossG", lossG, i)

            # 画图
            out = G.model.forward(bz=10).cpu().detach().numpy()
            showImg(out)

            if i % 1000 == 0:
                torch.save(D.model, DModelPath + "_" + str(i) + ".pkl")
            torch.save(G.model, GModelPath + "_" + str(i) + ".pkl")

            torch.save(D.model, DModelPath + ".pkl")
            torch.save(G.model, GModelPath + ".pkl")


def dev():
    GBatchSize = 16
    lr = 0.001
    device = "cuda"
    GModelPath = "./model/GModel.pkl"
    G = Generator(device=device, batch_size=GBatchSize, lr=lr, load_model_path=GModelPath)
    out = G.model.forward(GBatchSize).cpu().detach()
    showImg(out)


if __name__ == '__main__':
    # train()
    # test()
    for i in range(10):
        dev()
