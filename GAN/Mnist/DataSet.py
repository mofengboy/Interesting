import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
import itertools


def getMnistTrain(data_set, batch_size):
    trainSet = MnistDataSet(file_path=data_set, is_train=True)
    trainDataLoader = DataLoader(trainSet, batch_size=batch_size, shuffle=True)
    # 无限迭代
    return next(itertools.cycle(trainDataLoader))


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
