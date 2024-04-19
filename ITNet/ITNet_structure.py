# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 13:08:19 2020

@author: 77509
"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from FIR_filter import my_filter


class ITNet_pre(nn.Module):
    """
    ITNet_pre is The source domain branch of ITNet,
    which is used in pre-training step

    """
    def __init__(self,
                 n_class,
                 n_channel,
                 input_length,

                 n_spatial=12,
                 fir_list=[],
                 fir_length=51,

                 pool_size=100,
                 pool_stride=25, ):
        super(ITNet_pre, self).__init__()
        self.__dict__.update(locals())
        del self.self
        self.n_fir = len(fir_list)

        ''' network structure '''
        self.IDConv_s = nn.Conv2d(1, self.n_spatial, (1, self.n_channel), stride=(1, 1), bias=False, )
        self.TemporalConv = nn.Conv2d(1, self.n_fir, (self.fir_length, 1), stride=(1, 1), bias=False, )
        self.TemporalConv.weight.requires_grad = False

        self.BN = nn.BatchNorm2d(self.n_fir, momentum=0.1, affine=True, eps=1e-5, )
        self.poolmean = nn.AvgPool2d(kernel_size=(self.pool_size, 1), stride=(self.pool_stride, 1))

        self.DRConv = nn.Conv2d(self.n_fir,
                                30,
                                kernel_size=(1, self.n_spatial),
                                stride=(1, 1),
                                bias=False, )

        self.ClassConv = nn.Conv2d(30,
                                   self.n_class,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   bias=False, )

        self.softmax_log = nn.LogSoftmax(dim=1)

        ''' the kernels are initialized as FIR filers '''
        fil = np.zeros([self.n_fir, 1, self.fir_length, 1])
        for i, lh in enumerate(self.fir_list):
            fil[i, 0, :, 0] = my_filter(fs=250, fl=lh[0], fh=lh[1], length=51)
        fir_tensor = torch.Tensor(fil)
        self.TemporalConv.weight = torch.nn.Parameter(fir_tensor)

        ''' random initialization '''
        nn.init.xavier_uniform_(self.IDConv_s.weight, gain=1)
        nn.init.xavier_uniform_(self.ClassConv.weight, gain=1)
        nn.init.xavier_uniform_(self.DRConv.weight, gain=1)

        nn.init.constant_(self.BN.weight, 1)
        nn.init.constant_(self.BN.bias, 0)

    def forward(self, x):  # input size (b, 22, 875, 1)
        x = x.permute(0, 3, 2, 1)
        x = self.IDConv_s(x)
        x = x.permute(0, 3, 2, 1)
        x = self.TemporalConv(x)

        ''' feature extraction per sub-band '''
        x = self.BN(x)
        x = torch.mul(x, x)
        x = self.poolmean(x)

        x = self.DRConv(x)
        x = self.softmax_log(self.ClassConv(x))
        x = x.squeeze()
        return x


class ITNet(nn.Module):
    def __init__(self,
                 source_subject=1,
                 mode_loss = 'fpr',
                 n_class = 4,
                 n_channel = 22,
                 input_length = 875,

                 n_spatial=12,
                 fir_list=[],
                 fir_length=51,

                 pool_size=100,
                 pool_stride=25):

        super(ITNet, self).__init__()
        self.__dict__.update(locals())
        del self.self
        self.n_fir = len(fir_list)

        ''' network structure '''
        self.IDConv_s = nn.Conv2d(1, self.n_spatial, (1, self.n_channel), stride=(1, 1), bias=False, )
        self.IDConv_s.weight.requires_grad = False
        self.IDConv_t = nn.Conv2d(1, self.n_spatial, (1, self.n_channel), stride=(1, 1), bias=False, )
        self.TemporalConv = nn.Conv2d(1, self.n_fir, (self.fir_length, 1), stride=(1, 1), bias=False, )
        self.TemporalConv.weight.requires_grad = False

        self.TransConv = nn.Conv2d(self.n_fir, self.n_fir, (1, 1), (1, 1), bias=False)

        self.BN = nn.BatchNorm2d(self.n_fir, momentum=0.1, affine=True, eps=1e-5, )
        self.poolmean = nn.AvgPool2d(kernel_size=(self.pool_size, 1), stride=(self.pool_stride, 1))

        self.DRConv = nn.Conv2d(self.n_fir,
                                 30,
                                 kernel_size=(1, self.n_spatial),
                                 stride=(1, 1),
                                 bias=False, )

        self.ClassConv = nn.Conv2d(30,
                                    self.n_class,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    bias=False, )

        self.DRConv.weight.requires_grad = False
        self.ClassConv.weight.requires_grad = False

        self.softmax_log = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)

        self.train = True


        fil = np.zeros([self.n_fir, 1, self.fir_length, 1])
        for i, lh in enumerate(self.fir_list):
            fil[i, 0, :, 0] = my_filter(fs=250, fl=lh[0], fh=lh[1], length=51)
        fir_tensor = torch.Tensor(fil)
        self.TemporalConv.weight = torch.nn.Parameter(fir_tensor)
        
        w_trans = np.eye(self.n_fir).reshape([self.n_fir, self.n_fir, 1, 1]).astype('float32')
        self.TransConv.weight = torch.nn.Parameter(torch.tensor(w_trans))

        ''' use pre-training result to initialize'''
        path = 'pretrain/S%d/default/' % self.source_subject

        w_spatial = np.load(path+'w_id.npy')
        self.IDConv_s.weight = torch.nn.Parameter(torch.tensor(w_spatial, dtype=torch.float))
        self.IDConv_t.weight = torch.nn.Parameter(torch.tensor(w_spatial, dtype=torch.float))

        w_BN = np.load(path+'w_BN.npy')
        self.BN.weight = torch.nn.Parameter(torch.tensor(w_BN, dtype=torch.float))
        b_BN = np.load(path+'b_BN.npy')
        self.BN.bias = torch.nn.Parameter(torch.tensor(b_BN, dtype=torch.float))

        w_dr = np.load(path+'w_dr.npy')
        self.DRConv.weight = torch.nn.Parameter(torch.tensor(w_dr))
        w_class = np.load(path+'w_class.npy')
        self.ClassConv.weight = torch.nn.Parameter(torch.tensor(w_class))


    def forward(self, x_t, x_s):#
        x_s = x_t
        x_t = x_t.permute(0, 3, 2, 1)
        x_t = self.IDConv_t(x_t)
        x_t = x_t.permute(0, 3, 2, 1)
        x_t = self.TemporalConv(x_t)
        x_t_s = self.TransConv(x_t)

        if self.train:
            x_s = x_s.permute(0, 3, 2, 1)
            x_s = self.IDConv_s(x_s)
            x_s = x_s.permute(0, 3, 2, 1)
            x_s = self.TemporalConv(x_s)

            n_t = x_t.data.size(0)
            x = torch.cat((x_t_s, x_s), 0) # the two-branch data vectors are concatenated
            x = self.BN(x)
            x = torch.mul(x, x)
            x = self.poolmean(x)

            x_dr = self.DRConv(x[:n_t]) # dimension reduction
            if self.mode_loss == 'fpr':
                g = torch.sigmoid(self.softmax_log(x_dr)*3) # with activation function
            else:
                g = self.softmax(x_dr) # without activation function

            ''' 
                self.softmax_log include softmax and log calculations
                softmax is used to build constraint
                sigmoid(3*log(x)) = 1/(1+x^(-3))
                                                 '''  

            dis = F.pairwise_distance(g, g[:,:,torch.randperm(30)])
            dis = torch.mean(dis)

            x = self.ClassConv(x_dr)
            x = self.softmax_log(x)
            # print(x.data.size())

        else:
            x = self.BN(x_t_s)
            x = torch.mul(x, x)
            x = self.poolmean(x)
            x_dr = self.DRConv(x)
            x = self.softmax_log(self.ClassConv(x_dr))
            dis = 0

        x = x.squeeze()
        return x, dis


class ITNet_heavy(nn.Module):
    def __init__(self,
                 source_subject=1,
                 mode_loss = 'fpr',
                 n_class = 4,
                 n_channel = 22,
                 input_length = 875,

                 n_spatial=12,
                 fir_list=[],
                 fir_length=51,

                 pool_size=100,
                 pool_stride=25):

        super(ITNet_heavy, self).__init__()
        self.__dict__.update(locals())
        del self.self

        self.n_fir = len(fir_list)

        self.IDConv_s = nn.Conv2d(1, self.n_spatial, (1, self.n_channel), stride=(1, 1), bias=False, )
        self.IDConv_s.weight.requires_grad = False

        self.IDConv_t = nn.Conv2d(1, self.n_spatial, (1, self.n_channel), stride=(1, 1), bias=False, )


        self.TemporalConv_s = nn.Conv2d(1, self.n_fir, (self.fir_length, 1), stride=(1, 1), bias=False, )
        self.TemporalConv_s.weight.requires_grad = False
        self.TemporalConv_t = nn.Conv2d(1, self.n_fir, (self.fir_length, 1), stride=(1, 1), bias=False, )


        self.BN = nn.BatchNorm2d(self.n_fir, momentum=0.1, affine=True, eps=1e-5, )

        self.poolmean = nn.AvgPool2d(kernel_size=(self.pool_size, 1), stride=(self.pool_stride, 1))


        self.DRConv = nn.Conv2d(self.n_fir,
                                 30,
                                 kernel_size=(1, self.n_spatial),
                                 stride=(1, 1),
                                 bias=False, )

        self.ClassConv = nn.Conv2d(30,
                                    self.n_class,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    bias=False, )

        self.DRConv.weight.requires_grad = False
        self.ClassConv.weight.requires_grad = False

        self.softmax_log = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)

        self.train = True

        fil = np.zeros([self.n_fir, 1, self.fir_length, 1])
        for i, lh in enumerate(self.fir_list):
            fil[i, 0, :, 0] = my_filter(fs=250, fl=lh[0], fh=lh[1], length=51)
        fir_tensor = torch.Tensor(fil)
        self.TemporalConv_s.weight = torch.nn.Parameter(fir_tensor)
        self.TemporalConv_t.weight = torch.nn.Parameter(fir_tensor)


        path = 'pretrain/S%d/default/' % self.source_subject

        w_spatial = np.load(path+'w_id.npy')
        self.IDConv_s.weight = torch.nn.Parameter(torch.tensor(w_spatial, dtype=torch.float))
        self.IDConv_t.weight = torch.nn.Parameter(torch.tensor(w_spatial, dtype=torch.float))

        w_BN = np.load(path+'w_BN.npy')
        self.BN.weight = torch.nn.Parameter(torch.tensor(w_BN, dtype=torch.float))
        b_BN = np.load(path+'b_BN.npy')
        self.BN.bias = torch.nn.Parameter(torch.tensor(b_BN, dtype=torch.float))

        w_dr = np.load(path+'w_dr.npy')
        self.DRConv.weight = torch.nn.Parameter(torch.tensor(w_dr))
        w_class = np.load(path+'w_class.npy')
        self.ClassConv.weight = torch.nn.Parameter(torch.tensor(w_class))


    def forward(self, x_t, x_s):
        x_t = x_t.permute(0, 3, 2, 1)
        x_t = self.IDConv_t(x_t)
        x_t = x_t.permute(0, 3, 2, 1)
        x_t = self.TemporalConv_t(x_t)

        if self.train:
            x_s = x_s.permute(0, 3, 2, 1)
            x_s = self.IDConv_s(x_s)
            x_s = x_s.permute(0, 3, 2, 1)
            x_s = self.TemporalConv_s(x_s)

            n_t = x_t.data.size(0)
            x = torch.cat((x_t, x_s), 0)
            x = self.BN(x)
            x = torch.mul(x, x)
            x = self.poolmean(x)

            x_dr = self.DRConv(x[:n_t])
            if self.mode_loss == 'fpr':
                g = torch.sigmoid(self.softmax_log(x_dr)*3)
            else:
                g = self.softmax(x_dr)

            dis = F.pairwise_distance(g, g[:,:,torch.randperm(30)])
            dis = torch.mean(dis)

            x = self.ClassConv(x_dr)
            x = self.softmax_log(x)
            # print(x.data.size())

        else:
            x = self.BN(x_t)
            x = torch.mul(x, x)
            x = self.poolmean(x)
            x_dr = self.DRConv(x)
            x = self.softmax_log(self.ClassConv(x_dr))
            dis = 0

        x = x.squeeze()
        return x, dis

    
class dis(nn.Module):
    def __init__(self):
        super(dis, self).__init__()
        self.__dict__.update(locals())
        self.softmax_log = nn.LogSoftmax(dim=1)
    def forward(self, x_dr):
        #g = torch.sigmoid(self.softmax_log(x_dr)*3) # with activation function
        g = x_dr
        dis = F.pairwise_distance(g, g[:,:,torch.randperm(30)])
        dis = torch.mean(dis)
        return dis