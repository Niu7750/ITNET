import torch
import mne
import numpy as np
from torch.utils.data import DataLoader,Dataset,TensorDataset
import random

'''

build datasets in tensor format

'''

def my_dataset(xnp, ynp, batch_size, a):
    xnp = xnp.astype('float32')
    ynp = ynp.squeeze()

    x_train = torch.from_numpy(xnp)
    x_train = x_train.unsqueeze(3)  # increase dimension

    y_train = torch.from_numpy(ynp).long()

    dataset1 = TensorDataset(x_train, y_train)
    data_train = DataLoader(dataset=dataset1, batch_size=batch_size, shuffle=a)
    return data_train


'''source and target samples are combined to fed into ITNet'''
class Sample_couple(Dataset):
    def __init__(self, src_x, src_y, tgt_label_x, tgt_label_y):
        self.src_x = src_x.astype('float32')
        self.tgt_label_x = tgt_label_x.astype('float32')
        self.src_y = src_y
        self.tgt_label_y = tgt_label_y
        self.n_sample_s = src_x.shape[0]
        self.n_sample_t = tgt_label_x.shape[0]

    def __getitem__(self, index):
        n1 = random.randint(0, self.n_sample_t - 1)
        data_t = self.tgt_label_x[n1].reshape([22, 875, 1])  # apply 'reshape' to increase dimension

        for i in range(20):
            n2 = random.randint(0, self.n_sample_s - 1)
            if self.src_y[n2] == self.tgt_label_y[n1]:
                break

        data_s = self.src_x[n2].reshape([22, 875, 1])

        return data_t, data_s, self.src_y[n2].squeeze()

    def __len__(self):
        return self.n_sample_s
