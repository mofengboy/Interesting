# 使用生成对抗网络（GAN）生成手写体

### 原理简述
GAN由两个子网络构成，一个鉴别网络，一个生成网络，大多数情况下我们需要的是生成网络。
训练过程中，这两个网络相辅相成，相互对抗，一起学习。

鉴别网络用来执行二分类任务，目标是区分出输入自己的数据是来自于真实数据集还是生成网络生成的数据集。

生成网络用来执行生成任务，目标是生成尽可能和真实数据集同分布的数据，使鉴别网络无法正确区分出数据的来源。

### 数据读取部分
基于Dataset类构建，并实现其中的三个方法。Dataset类位于torch.utils.data
```python
class MnistDataSet(Dataset):
    def __init__(self, file_path, is_train=True):
        # 使用scipy读入mat文件数据
        mnist_all = sio.loadmat(file_path)
        train_raw = []
        test_raw = []
        # 依次读入数据集0-9
        for i in range(10):
            train_temp = mnist_all["train" + str(i)]
            for j in train_temp:
                j = np.array(j) / 255.0
                train_raw.append([j, i])
        for i in range(10):
            test_temp = mnist_all["test" + str(i)]
            for j in test_temp:
                j = np.array(j) / 255.0
                test_raw.append([j, i])

        self.trainDataSet = train_raw
        self.testDataSet = test_raw
        self.is_train = is_train

    def __getitem__(self, index):
        if self.is_train:
            dataSet = self.trainDataSet
        else:
            dataSet = self.testDataSet
        img = dataSet[index][0]
        labelArr = np.eye(10)
        label = labelArr[dataSet[index][1]]
        return img, label

    def __len__(self):
        if self.is_train:
            return len(self.trainDataSet)
        else:
            return len(self.testDataSet)
```

因为整个训练过程中需要不断的读取数据，所以在这里将数据打乱之后，实现了一个无限迭代器。

```python
def getMnistTrain(data_set, batch_size):
    trainSet = MnistDataSet(file_path=data_set, is_train=True)
    trainDataLoader = DataLoader(trainSet, batch_size=batch_size, shuffle=True)
    # 无限迭代
    return next(itertools.cycle(trainDataLoader))
```
### 鉴别网络
鉴别网络类定义中有一个初始化方法，一个Train方法和一个卷积+全连接的二分类网络子类。

```python
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
```

### 生成网络
生成网络的基本架构和鉴别网络相同，一个类初始化方法，一个train方法,一个模型子类。

在生成网络中的生成子模型的构建是参与全连接+上采样+卷积，这里最开始我用的是全连接+反卷积，但是有资料说不如上采样加卷积效果好。

在初始化生成模型时，若指定load_model_path参数，则会加载指定模型，方便后续生成或继续训练。

```python
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
```
### 训练
训练主要分为三个步骤。

1. 用真实数据集训练鉴别网络
2. 用生成网络生成的数据训练鉴别网络
3. 以鉴别网络作为参考训练生成网络

实际训练过程中，鉴别网络比生成网络学习的速度快多了，所以为了保证一方不处于绝对优势，鉴别网络的在每一轮的学习次数和Batch_size参数小于生成网络。

```python
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
```
### 验证
我在Calab上训练了6000次，由于只保存了模型在谷歌云端硬盘上，第二天起来log记录没了，所以只放一下生成的图片就不放选了过程中的loss图了。

```python
def dev():
    GBatchSize = 16
    lr = 0.001
    device = "cuda"
    GModelPath = "./model/GModel.pkl"
    G = Generator(device=device, batch_size=GBatchSize, lr=lr, load_model_path=GModelPath)
    out = G.model.forward(GBatchSize).cpu().detach()
    showImg(out)
```

![mark](https://external-link.sunan.me/blog/210527/LB8ddHi92f.png?imageslim)
![mark](https://external-link.sunan.me/blog/210527/bHif69CDdm.png?imageslim)
![mark](https://external-link.sunan.me/blog/210527/7h2Fe5mEeF.png?imageslim)

### 源代码

> [https://github.com/mofengboy/Interesting]