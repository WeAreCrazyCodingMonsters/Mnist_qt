from torchvision import datasets, transforms
import torchvision
import os
from skimage import io
import torchvision.datasets.mnist as mnist

import config
from config import get_config


#
# config = get_config('D:\Project\mnist_attack\configs\configMNIST.yaml')
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# train_data = torchvision.datasets.FashionMNIST(
#     root='../dataset',  # 数据集的位置
#     train=True,  # 如果为True则为训练集，如果为False则为测试集
#     transform=torchvision.transforms.ToTensor(),  # 将图片转化成取值[0,1]的Tensor用于网络处理
#     download=False
# )
# test_data = torchvision.datasets.FashionMNIST(
#     root='../dataset',  # 数据集的位置
#     train=False,  # 如果为True则为训练集，如果为False则为测试集
#     transform=torchvision.transforms.ToTensor(),  # 将图片转化成取值[0,1]的Tensor用于网络处理
#     download=False
# )
#

#

def convert_to_img(train=True, root=None, train_set=None, test_set=None):
    if train:
        f = open('dataset/' + 'train_{}.txt'.format('ui'), 'w')
        data_path = root + '/train/'
        if (not os.path.exists(data_path)):
            os.makedirs(data_path)
        for i, (img, label) in enumerate(zip(train_set[0], train_set[1])):
            img_path = data_path + str(i) + '.jpg'
            io.imsave(img_path, img.numpy())
            f.write(img_path[1:] + ' ' + str(label.item()) + '\n')
        f.close()
    else:
        f = open('dataset/' + 'test.txt{}.txt'.format('ui'), 'w')
        data_path = root + '/test/'
        if (not os.path.exists(data_path)):
            os.makedirs(data_path)
        for i, (img, label) in enumerate(zip(test_set[0], test_set[1])):
            img_path = data_path + str(i) + '.jpg'
            io.imsave(img_path, img.numpy())
            f.write(img_path[1:] + ' ' + str(label.item()) + '\n')
        f.close()


def get_list(root_dir=None):
    train_set = (
        mnist.read_image_file(os.path.join(root_dir, 'train-images-idx3-ubyte')),
        mnist.read_label_file(os.path.join(root_dir, 'train-labels-idx1-ubyte'))
    )
    test_set = (
        mnist.read_image_file(os.path.join(root_dir, 't10k-images-idx3-ubyte')),
        mnist.read_label_file(os.path.join(root_dir, 't10k-labels-idx1-ubyte'))
    )
    print("training set :", train_set[0].size())
    print("test set :", test_set[0].size())

    convert_to_img(True, root=root_dir, train_set=train_set, test_set=test_set)  # 转换训练集
    convert_to_img(False, root=root_dir, train_set=train_set, test_set=test_set)  # 转换测试集
