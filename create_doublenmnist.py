#!/bin/python
#-----------------------------------------------------------------------------
# File Name : test_doublenmnist.py
# Author: Emre Neftci
#
# Creation Date : Thu Nov  7 20:30:14 2019
# Last Modified : 
#
# Copyright : (c) UC Regents, Emre Neftci
# Licence : GPLv2
#----------------------------------------------------------------------------- 
from torchneuromorphic.doublenmnist.doublenmnist_dataloaders import *

def create_datasets(
        root = 'data/nmnist/n_mnist.hdf5',
        batch_size = 72 ,
        chunk_size_train = 300,
        chunk_size_test = 300,
        ds = 1,
        dt = 1000,
        transform_train = None,
        transform_test = None,
        target_transform_train = None,
        target_transform_test = None,
        nclasses = 5,
        samples_per_class = 2,
        classes_meta = np.arange(100, dtype='int')):

    size = [2, 32//ds, 32//ds]

    if transform_train is None:
        transform_train = Compose([
            CropDims(low_crop=[0,0], high_crop=[32,32], dims=[2,3]),
            Downsample(factor=[dt,1,ds,ds]),
            ToEventSum(T = chunk_size_train, size = size),
            ToTensor()])
    if transform_test is None:
        transform_test = Compose([
            CropDims(low_crop=[0,0], high_crop=[32,32], dims=[2,3]),
            Downsample(factor=[dt,1,ds,ds]),
            ToEventSum(T = chunk_size_test, size = size),
            ToTensor()])
    if target_transform_train is None:
        target_transform_train =Compose([Repeat(chunk_size_train)])
    if target_transform_test is None:
        target_transform_test = Compose([Repeat(chunk_size_test)])


    labels_u = np.random.choice(classes_meta, nclasses, replace = False ) #100 here becuase we have two pairs of digits between 0 and 9

    train_ds = DoubleNMNISTDataset(root,train=True,
                                 transform = transform_train, 
                                 target_transform = target_transform_train, 
                                 chunk_size = chunk_size_train,
                                 nclasses = nclasses,
                                 samples_per_class = samples_per_class,
                                 labels_u = labels_u)

    test_ds = DoubleNMNISTDataset(root, transform = transform_test, 
                                 target_transform = target_transform_test, 
                                 train=False,
                                 chunk_size = chunk_size_test,
                                 nclasses = nclasses,
                                 samples_per_class = samples_per_class,
                                 labels_u = labels_u)

    return train_ds, test_ds


def create_dataloader(
        root = 'data/nmnist/n_mnist.hdf5',
        batch_size = 72 ,
        chunk_size_train = 300,
        chunk_size_test = 300,
        ds = 1,
        dt = 1000,
        transform_train = None,
        transform_test = None,
        target_transform_train = None,
        target_transform_test = None,
        nclasses = 5,
        samples_per_class = 2,
        classes_meta = np.arange(100, dtype='int'),
        **dl_kwargs):


    train_d, test_d = create_datasets(
        root = 'data/nmnist/n_mnist.hdf5',
        batch_size = batch_size,
        chunk_size_train = chunk_size_train,
        chunk_size_test = chunk_size_test,
        ds = ds,
        dt = dt,
        transform_train = transform_train,
        transform_test = transform_test,
        target_transform_train = target_transform_train,
        target_transform_test = target_transform_test,
        classes_meta = classes_meta,
        nclasses = nclasses,
        samples_per_class = samples_per_class)


    train_dl = torch.utils.data.DataLoader(train_d, shuffle=True, batch_size=batch_size, **dl_kwargs)
    test_dl = torch.utils.data.DataLoader(test_d, shuffle=False, batch_size=batch_size, **dl_kwargs)

    return train_dl, test_dl


def sample_double_mnist_task( N = 5,
                              K = 2,
                              meta_split = [range(64), range(64,80), range(80,100)],
                              meta_dataset_type = 'train',
                              **kwargs):
    classes_meta = {}
    classes_meta['train'] = np.array(meta_split[0], dtype='int')
    classes_meta['val']   = np.array(meta_split[1], dtype='int')
    classes_meta['test']  = np.array(meta_split[2], dtype='int')

    assert meta_dataset_type in ['train', 'val', 'test']
    return create_dataloader(classes_meta = classes_meta[meta_dataset_type], nclasses= N, samples_per_class = K, **kwargs)

def create_doublenmnist(batch_size = 50, shots=10, ways=5, mtype='train', **kwargs):
    train_dl, test_dl = sample_double_mnist_task(
            meta_dataset_type = mtype,
            N = ways,
            K = shots,
            root='data/nmnist/n_mnist.hdf5',
            batch_size=batch_size,
            num_workers=4,
            **kwargs)
    return train_dl, test_dl


if __name__ == '__main__':
    trdl, tedl = create_doublenmnist(ds=2)
