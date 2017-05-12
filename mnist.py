#!/usr/bin/python3

from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range

class Dataset(object):
    def __init__(self,images,labels,one_hot=False):
        self.images = images.astype(np.float32)
        if one_hot:
            self.labels = self.one_hot_transform(labels)
        else:
            self.labels = labels
        self.epochs_completed = 0
        self.index_in_epoch = 0
        self.data_length = images.shape[0]
    
    def one_hot_transform(self,labels):
        label_length = labels.shape[0]
        one_hot_labels = np.zeros( (label_length,10), dtype=np.float32)
        for i in range(label_length):
            one_hot_labels[i][labels[i]] = 1.0
        return one_hot_labels

    def next_batch(self,batch_size):
        start = self.index_in_epoch
        self.index_in_epoch += batch_size
        # handle corner cases for data depletion
        if self.index_in_epoch > self.data_length:
            self.epochs_completed += 1
            perm = np.arange(self.data_length)
            np.random.shuffle(perm)
            self.images = self.images[perm]
            self.labels = self.labels[perm]
            start = 0
            self.index_in_epoch = batch_size
            
        end = self.index_in_epoch
        return self.images[start:end],self.labels[start:end]
    

class MnistData(object):
    def __init__(self, pickle_file, one_hot=False):
        with open(pickle_file,'rb') as f:
            saved = pickle.load(f)
            self.train_data = Dataset(saved['train_dataset'],saved['train_labels'],one_hot)
            self.valid_data = Dataset(saved['valid_dataset'],saved['valid_labels'],one_hot)
            self.test_data = Dataset(saved['test_dataset'],saved['test_labels'],one_hot)
            del saved



            
