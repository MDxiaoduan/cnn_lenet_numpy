import numpy as np
import struct
import cv2
import matplotlib.pyplot as plt
from ifunction import Batch_Normalization

CHAR = "0123456789"


def loadImageSet(binfile):
    buffers = binfile.read()
    head = struct.unpack_from('>IIII', buffers, 0)
    offset = struct.calcsize('>IIII')
    imgNum = head[1]
    width = head[2]
    height = head[3]
    # [60000]*28*28
    bits = imgNum * width * height
    bitsString = '>' + str(bits) + 'B'  # like '>47040000B'
    imgs = struct.unpack_from(bitsString, buffers, offset)
    binfile.close()
    imgs = np.reshape(imgs, [imgNum, 1, width * height])
    return imgs


def loadLabelSet(binfile):
    buffers = binfile.read()
    head = struct.unpack_from('>II', buffers, 0)
    imgNum = head[1]
    offset = struct.calcsize('>II')
    numString = '>' + str(imgNum) + "B"
    labels = struct.unpack_from(numString, buffers, offset)
    binfile.close()
    labels = np.reshape(labels, [imgNum, 1])
    return labels


class mnist:
    def __init__(self, batch_size):
        self.train_images_in = open("MNIST_data\\train-images.idx3-ubyte", 'rb')
        self.train_labels_in = open("MNIST_data\\train-labels.idx1-ubyte", 'rb')
        self.test_images_in = open("MNIST_data\\t10k-images.idx3-ubyte", 'rb')
        self.test_labels_in = open("MNIST_data\\t10k-labels.idx1-ubyte", 'rb')
        self.batch_size = batch_size
        self.train_image = loadImageSet(self.train_images_in)                            # [60000, 1, 784]
        self.train_labels = loadLabelSet(self.train_labels_in)                           # [60000, 1]
        self.test_images = loadImageSet(self.test_images_in)                             # [10000, 1, 784]
        self.test_labels = loadLabelSet(self.test_labels_in)                             # [10000, 1]
        self.data = {"train": self.train_image, "test": self.test_images}
        self.label = {"train": self.train_labels, "test": self.test_labels}
        self.indexes = {"train": 0, "val": 0, "test": 0}

    def get_mini_bath(self, data_name="train"):
        if self.indexes[data_name]*self.batch_size > self.data[data_name].shape[0]:
            self.indexes[data_name] = 0
        batch_data = self.data[data_name][self.indexes[data_name]*self.batch_size:(self.indexes[data_name]+1)*self.batch_size, :, :]
        batch_label = self.label[data_name][self.indexes[data_name]*self.batch_size:(self.indexes[data_name]+1)*self.batch_size, :]
        self.indexes[data_name] += 1
        y = np.zeros((self.batch_size, len(CHAR)))
        for kk in range(self.batch_size):
            y[kk, CHAR.index(str(int(batch_label[kk])))] = 1.0
        x = Batch_Normalization(batch_data)
        return x, y


class cifar10:
    def load_cifar10(self, data_filename):
        """
        每个文件都是# image:(50000, 32, 32, 3) label:(50000,)大小，image是0-255没有做归一化的数据，label是0.0-9.0的数字
        "D:\\ipython\\data\\cifar10\\data\\CIFAR10_train_image.npy"      # [50000, 32, 32, 3]
        "D:\\ipython\\data\\cifar10\\data\\CIFAR10_train_label.npy"      # [50000]
        "D:\\ipython\\data\\cifar10\\data\\CIFAR10_test_image.npy"       # [10000, 32, 32, 3]
        "D:\\ipython\\data\\cifar10\\data\\CIFAR10_test_label.npy"       # [10000]
        """
        data = np.load(data_filename)
        L = data.shape[0]
        # BGR 是图片三个通道默认排列顺序 Gray = 0.3R + 0.59G + 0.11B
        data_Gray = 0.3 * data[:, :, :, 2] + 0.59 * data[:, :, :, 1] + 0.11 * data[:, :, :, 0]
        out_data = np.zeros((L, 28, 28))
        for kk in range(L):
            out_data[kk, :, :] = data_Gray[kk, 2:30, 2:30]  # 必须循环，三维矩阵直接np.resize会改变图像
        # normalization
        mean = np.mean(out_data)
        std = np.std(out_data)
        image_data = (out_data - mean) / std
        return image_data


def plot_images(images, labels):
    for i in np.arange(0, 16):
        plt.subplot(4, 4, i + 1)
        plt.axis('off')
        plt.title(labels[i], fontsize=14)
        plt.subplots_adjust(top=1.5)
        plt.imshow(np.reshape(images[i], (28, 28)), cmap='gray')
    plt.show()

# if __name__ == "__main__":
#     mnist = mnist(batch_size=16)
#     x, y = mnist.get_mini_bath(data_name="test")
#     plot_images(x, y)              # [16, 1, 784]  [16, 1]
