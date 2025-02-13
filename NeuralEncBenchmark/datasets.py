import os
import numpy as np
import torch
import torchvision as tv

def rgb2gray(rgb):
    r, g, b = rgb[:,:,:,0], rgb[:,:,:,1], rgb[:,:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def load_mnist():
    root_mnist = os.path.expanduser("~/data/datasets/torch/mnist")
    
    train_mnist_dataset = tv.datasets.MNIST(root_mnist, train=True, transform=None,
                                                       target_transform=None, download=True)
    test_mnist_dataset = tv.datasets.MNIST(root_mnist, train=False, transform=None, 
                                                      target_transform=None, download=True)
    
    x_train_mnist = np.array(train_mnist_dataset.data, dtype=np.float)
    x_train_mnist = x_train_mnist.reshape(x_train_mnist.shape[0],-1)/255
    
    x_test_mnist = np.array(test_mnist_dataset.data, dtype=np.float)
    x_test_mnist = x_test_mnist.reshape(x_test_mnist.shape[0],-1)/255
    
    y_train_mnist = np.array(train_mnist_dataset.targets, dtype=np.int)
    y_test_mnist  = np.array(test_mnist_dataset.targets, dtype=np.int)
    
    
    return {'x_train': x_train_mnist, 'x_test': x_test_mnist,
            'y_train': y_train_mnist, 'y_test':y_test_mnist,
            'train_dataset': train_mnist_dataset, 'test_dataset': test_mnist_dataset}

def load_fmnist():
    root_fmnist = os.path.expanduser("~/data/datasets/torch/fashion-mnist")
    train_fmnist_dataset = tv.datasets.FashionMNIST(root_fmnist, train=True, transform=None, target_transform=None, download=True)
    test_fmnist_dataset = tv.datasets.FashionMNIST(root_fmnist, train=False, transform=None, target_transform=None, download=True)
    
    x_train_fmnist = np.array(train_fmnist_dataset.data, dtype=np.float)
    x_train_fmnist = x_train_fmnist.reshape(x_train_fmnist.shape[0],-1)/255
    
    x_test_fmnist = np.array(test_fmnist_dataset.data, dtype=np.float)
    x_test_fmnist = x_test_fmnist.reshape(x_test_fmnist.shape[0],-1)/255
    
    y_train_fmnist = np.array(train_fmnist_dataset.targets, dtype=np.int)
    y_test_fmnist  = np.array(test_fmnist_dataset.targets, dtype=np.int)    
    
    return {'x_train': x_train_fmnist, 'x_test': x_test_fmnist,
            'y_train': y_train_fmnist, 'y_test':y_test_fmnist,
            'train_dataset': train_fmnist_dataset, 'test_dataset': test_fmnist_dataset}

def load_cifar10_gray():
    root_cifar = os.path.expanduser("~/data/datasets/torch/cifar")
    
    train_cifar_dataset = tv.datasets.CIFAR10(root_cifar, train=True, transform=None,
                                                       target_transform=None, download=True)
    test_cifar_dataset = tv.datasets.CIFAR10(root_cifar, train=False, transform=None, 
                                                      target_transform=None, download=True)
    
    x_train_cifar = np.array(rgb2gray(train_cifar_dataset.data), dtype=np.float)
    x_train_cifar = x_train_cifar.reshape(x_train_cifar.shape[0],-1)/255
    
    x_test_cifar = np.array(rgb2gray(test_cifar_dataset.data), dtype=np.float)
    x_test_cifar = x_test_cifar.reshape(x_test_cifar.shape[0],-1)/255
    
    y_train_cifar = np.array(train_cifar_dataset.targets, dtype=np.int)
    y_test_cifar  = np.array(test_cifar_dataset.targets, dtype=np.int)
    
    return {'x_train': x_train_cifar, 'x_test': x_test_cifar,
            'y_train': y_train_cifar, 'y_test':y_test_cifar,
            'train_dataset': train_cifar_dataset, 'test_dataset': test_cifar_dataset}
    

if __name__ == '__main__':
    load_mnist()