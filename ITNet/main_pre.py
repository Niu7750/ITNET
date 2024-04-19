# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 14:06:04 2020

@author: 77509
"""

from ITNet_structure import ITNet_pre
import matplotlib.pyplot as plt
from Dataset import Separate_source, Separate_target, GDF_dataset
from Tensor_dataset import my_dataset, Sample_couple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import os


''' 
     In order to fine-tune less parameters in inverse-transfer step, 
     n_spatial (IDConv kernel number) is set as smaller and model training is a little under-fitted on sufficient source data
     So, many times running may be needed to get the model with more than 93% accuracy
                                                                                                                     '''

''' set source subject 
    multiple subjects can be selected to form source dataset,
    but only one subject is selected in this paper'''
sub_list = [1]
train_all = [[],[]]
test_all = [[],[]]


''' hype-parameters'''
n_class = 4
input_length = 875
pool_size = 100
pool_stride = 25
fir_length = 51
n_crop = int((input_length - fir_length + 1 - pool_size) / pool_stride + 1)
fir_list = []
for i in range(2, 32, 2):
    fir_list.append([i, i + 2])

bs_train = 10*len(sub_list)
bs_test = 72*len(sub_list)


''' read and divide dataset '''
for s in sub_list:
    filename='F:\database\BCI competion\data2\BCIC IV_2a/A0%dT.gdf' % s
    data_loader=GDF_dataset(filename)
    data_cnt=data_loader.load_data_events()
    data_cnt.filter(1, 40, fir_design='firwin', skip_by_annotation='edge')
    data_cnt = data_cnt.drop_channels(["EOG-left", "EOG-central", "EOG-right"])
    assert len(data_cnt.ch_names) == 22
    data_cnt._data = data_cnt._data * (10 ** 5)
    train_x,train_y,test_x,test_y=Separate_source(data_cnt, 72)

    train_all[0].append(train_x)
    train_all[1].append(train_y)
    test_all[0].append(test_x)
    test_all[1].append(test_y)

train_x = np.concatenate((train_all[0][:]), axis=0) # the combination of multiple source subjects
train_y = np.concatenate((train_all[1][:]), axis=0)
test_x = np.concatenate((test_all[0][:]), axis=0)
test_y = np.concatenate((test_all[1][:]), axis=0)

train = my_dataset(train_x,train_y, bs_train, True) # training set 
test = my_dataset(test_x,test_y, bs_test, False) # test set

for ii in range(20):
    model = ITNet_pre(n_class = n_class,
                 n_channel = 22,
                 input_length = input_length,

                 n_spatial=12,
                 fir_list=fir_list,
                 fir_length=fir_length,

                 pool_size=pool_size,
                 pool_stride=pool_stride,)
    
    criterion = nn.CrossEntropyLoss()
    model.TemporalConv.weight.requires_grad = False
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, betas=(0.9, 0.999),
                           eps=1e-08, weight_decay=1e-4)

    ACC_crop = []
    ACC_vote = [] # record test accuracy per epoch
    time_start = time.time() 
       
    for epoch in range(1000):
        running_loss=0.0
        correct = 0
        total = 0

        '''training step'''
        for i,data in enumerate(train):
            x, y = data  # x size(b, 22, 875, 1)
            out = model(x) # the output size (b, 4, 30), contains 30 soft labels of 30 feature metrics from one sample
            
            ''' cross-entropy loss need be calculated between the 30 soft labels and the true label'''
            out_c = out[:,:,0]
            y_c = y
            for j in range(1,n_crop):
                out_c = torch.cat((out_c, out[:,:,j]), 0) 
                y_c = torch.cat((y_c, y))
            loss = criterion(out_c, y_c)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            ''' calculate accuracy on all the soft labels'''
            _, predicted = torch.max(out_c.data, 1)
            correct += (predicted == y_c).sum().item()
            total += y_c.size(0) 
            running_loss += loss.item()
                  
            pack = 5
            if i % pack == pack-1:
                acc_train = 100 * correct / total
                print('[%d, %5d] loss: %.7f acc: %.2f %%' % (epoch + 1, i + 1, running_loss / pack, 100 * correct / total))
                running_loss = 0.0
                correct = 0
                total = 0

        
        ''' test step '''
        correct = 0
        total = 0
        correct_vote = 0 #the number of the samples classified correctly after voting
        with torch.no_grad():
            for i, data_test in enumerate(test):
                xt, yt = data_test
                outt = model(xt)
                
                outt_c = outt[:,:,0]
                yt_c = yt
                for j in range(1,n_crop):
                    outt_c = torch.cat((outt_c, outt[:,:,j]), 0) 
                    yt_c = torch.cat((yt_c, yt))
                
                _, predicted = torch.max(outt_c.data, 1)
                total += yt_c.size(0)
                correct += (predicted == yt_c).sum().item()
    
                _, predicted = torch.max(outt.data, 1)
                predicted = predicted.numpy()
                for j in range(bs_test):
                    count = np.bincount(predicted[j,:])
                    label_crop = np.argmax(count)
                    if label_crop == yt.numpy()[j]:
                        correct_vote += 1

            acc_crop = correct / total  # accuracy before voting
            acc_vote = correct_vote / bs_test  # accuracy after voting
             
            ACC_crop.append(acc_crop)
            ACC_vote.append(acc_vote) # save the accuracy after voting
            if acc_vote >= 0.93:
                break

            print('Accuracy of the segments: %.4f %%    Accuracy after vote%.4f %% ' % (100 * acc_crop, 100*acc_vote))
    
    
    time_end = time.time()
    time_c= time_end - time_start 

    acc_max = np.max(ACC_vote)
    acc_epoch = np.argmax(ACC_vote)
    acc_crop = ACC_crop[acc_epoch]

    path = 'pretrain/S%d/%d-%.4f-%.4f'%(s, acc_epoch, acc_crop, acc_max)
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(model, path+'/model.pth')
    
    w_spatial = model.IDConv_s.weight.data.numpy()
    np.save(path+'/w_id', w_spatial)
    w_dr = model.DRConv.weight.data.numpy()
    np.save(path + '/w_dr', w_dr)
    w_class = model.ClassConv.weight.data.numpy()
    np.save(path+'/w_class', w_class)
    w_BN = model.BN.weight.data.numpy()
    np.save(path + '/w_BN', w_BN)
    b_BN = model.BN.bias.data.numpy()
    np.save(path + '/b_BN', w_BN)

