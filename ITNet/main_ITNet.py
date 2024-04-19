# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 14:06:04 2020

@author: 77509
"""

from ITNet_structure import ITNet
from Dataset import Separate_source, Separate_target, GDF_dataset
from Tensor_dataset import my_dataset, Sample_couple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Assume that we are on a CUDA machine, then this should print a CUDA device:
print(device)


seed_list = [[30, 38, 13, 50, 61],
             [7, 11, 10, 46, 21],
             [23, 15, 57, 42, 30],
             [],
             [7, 11, 10, 46, 21],
             [53, 36, 65, 28, 56],
             [7, 11, 10, 46, 21],
             [7, 11, 10, 46, 21],
             [61, 35, 18, 70, 14]] # random seed


""" * * * * * * * * * * * * * 
           please
            set  
            mode
* * * * * * * * * * * * * """

mode_list = ['none', 'fpr', 'linear']
mode_loss = mode_list[0]
"""
none : self-align ITNet
fpr : FPR-ITNet
linear : linear FPR-ITNet
"""

n_class = 4
input_length = 875
pool_size = 100
pool_stride = 25
fir_length = 51
n_crop = int((input_length-fir_length+1 - pool_size) / pool_stride + 1)
fir_list = []
for i in range(2, 32, 2):
    fir_list.append([i, i + 2])

bs_train = 8
bs_label = 20
bs_test = 288-bs_label

s_s = 1
for s_t in [7]:
    filename='F:\database\BCI competion\data2\BCIC IV_2a/A0%dT.gdf' % s_s
    data_loader=GDF_dataset(filename)
    data_cnt=data_loader.load_data_events()
    data_cnt.filter(1, 40, fir_design='firwin', skip_by_annotation='edge')
    data_cnt = data_cnt.drop_channels(["EOG-left", "EOG-central", "EOG-right"])
    assert len(data_cnt.ch_names) == 22
    data_cnt._data = data_cnt._data*(10**5)
    src_x, src_y, _, _ = Separate_source(data_cnt, 0)


    filename = 'F:\database\BCI competion\data2\BCIC IV_2a/A0%dT.gdf' % s_t
    data_loader = GDF_dataset(filename)
    data_cnt = data_loader.load_data_events()
    data_cnt.filter(1, 40, fir_design='firwin', skip_by_annotation='edge')
    data_cnt = data_cnt.drop_channels(["EOG-left", "EOG-central", "EOG-right"])
    assert len(data_cnt.ch_names) == 22
    data_cnt._data = data_cnt._data * (10 ** 5)


    for folder in range(4):
        for ii in range(20):
            tgt_x,tgt_y,label_x,label_y=Separate_target(data_cnt, bs_label, seed_list[s_t-1])
            tgt_label = my_dataset(label_x,label_y, bs_label, False)
            test = my_dataset(tgt_x,tgt_y, bs_test, False)

            iterator = Sample_couple(src_x, src_y, label_x, label_y)
            train =  DataLoader(iterator, shuffle=True, batch_size = bs_train)

            model = ITNet(
                 source_subject= s_s,
                 mode_loss=mode_loss,

                 n_class = n_class,
                 n_channel = 22,
                 input_length = input_length,

                 n_spatial=12,
                 fir_list=fir_list,
                 fir_length=fir_length,

                 pool_size=pool_size,
                 pool_stride=pool_stride,)

            model = model.to(device)

            criterion = nn.CrossEntropyLoss()

            model.IDConv_s.weight.requires_grad = False
            model.TemporalConv.weight.requires_grad = False
            model.ClassConv.weight.requires_grad = False
            model.DRConv.weight.requires_grad = False

            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0005, betas=(0.9, 0.999),
                                   eps=1e-08, weight_decay=1e-4)

            ACC_crop = []
            ACC_vote = []
            time_start = time.time()
            baseline = torch.ones(bs_label, 22, 875, 1).to(device)

            for epoch in range(400):
                Loss_class = 0.0
                Loss_dis = 0.0
                correct = 0
                total = 0

                model.train = True
                for i,data in enumerate(train):
                    x_t, x_s, y = data
                    x_t, x_s, y = x_t.to(device), x_s.to(device), y.to(device)
                    out, loss_dis = model(x_t, x_s)

                    out_c = out[:,:,0]
                    y = y.long()
                    y_c = y
                    for j in range(1,n_crop):
                        out_c = torch.cat((out_c, out[:,:,j]), 0)
                        y_c = torch.cat((y_c, y))

                    if mode_loss == 'none':
                        loss = criterion(out_c, y_c)
                    else:
                        loss = criterion(out_c, y_c) - loss_dis

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    _, predicted = torch.max(out_c.data, 1)

                    correct += (predicted == y_c).sum().item()
                    total += y_c.size(0)
                    Loss_class += criterion(out_c, y_c).item()
                    Loss_dis += loss_dis.item()

                    pack = 5
                    if i % pack == pack-1:
                        acc_train = 100 * correct / total
                        print('[%d, %5d] loss_class: %.7f loss_distance: %.7f acc: %.2f %%' % (epoch + 1, i + 1, Loss_class / pack, Loss_dis / pack, acc_train))
                        running_loss1 = 0.0
                        running_loss2 = 0.0
                        correct = 0
                        total = 0

                correct = 0
                total = 0
                correct_vote = 0
                with torch.no_grad():
                    model.train = False
                    for i,data in enumerate(tgt_label):
                        x, y = data
                        x, y = x.to(device), y.to(device)
                        out, _ = model(x, baseline)

                        out_c = out[:,:,0]
                        y_c = y
                        for j in range(1,n_crop):
                            out_c = torch.cat((out_c, out[:,:,j]), 0)
                            y_c = torch.cat((y_c, y))

                        _, predicted = torch.max(out_c.data, 1)
                        total += y_c.size(0)
                        correct += (predicted == y_c).sum().item()

                        _, predicted = torch.max(out.data, 1)
                        predicted = predicted.cpu().numpy()
                        y = y.cpu().numpy()
                        for j in range(bs_label):
                            count = np.bincount(predicted[j,:])
                            label_crop = np.argmax(count)
                            if label_crop==y[j]:
                                correct_vote += 1

                    acc_crop = correct / total
                    acc_vote = correct_vote/bs_label

                    print('Accuracy of the segments: %.4f %%    Accuracy after vote%.4f %% ' % (100*acc_crop, 100*acc_vote))


                correct = 0
                total = 0
                correct_vote = 0
                with torch.no_grad():
                    model.train = False
                    for i,data_test in enumerate(test):
                        xt, yt = data_test
                        xt, yt =  xt.to(device), yt.to(device)
                        outt, _ = model(xt, baseline)

                        outt_c = outt[:,:,0]
                        yt_c = yt
                        for j in range(1,n_crop):
                            outt_c = torch.cat((outt_c, outt[:,:,j]), 0)
                            yt_c = torch.cat((yt_c, yt))

                        _, predicted = torch.max(outt_c.data, 1)
                        total += yt_c.size(0)
                        correct += (predicted == yt_c).sum().item()

                        _, predicted = torch.max(outt.data, 1)
                        predicted = predicted.cpu().numpy()
                        yt = yt.cpu().numpy()
                        for j in range(bs_test):
                            count = np.bincount(predicted[j,:])
                            label_crop = np.argmax(count)
                            if label_crop==yt[j]:
                                correct_vote += 1

                    acc_crop = correct / total
                    acc_vote = correct_vote/bs_test

                    ACC_crop.append(acc_crop)
                    ACC_vote.append(acc_vote)
                    if acc_vote >= 1:
                        break

                    print('Accuracy of the segments: %.4f %%    Accuracy after vote%.4f %% ' % (100 * acc_crop, 100*acc_vote))

            time_end = time.time() 
            time_c= time_end - time_start 

            acc_max = np.max(ACC_vote)
            acc_epoch = np.argmax(ACC_vote)
            acc_crop = ACC_crop[acc_epoch]

            path = 'result/ITNet %s/S%d-%d(%d)/%d-%.4f-%.4f/'%(mode_loss, s_s, s_t, folder, acc_epoch, acc_crop, acc_max)
            if not os.path.exists(path):
                os.makedirs(path)
            torch.save(model, path+'/model.pth')

            w_spatial = model.IDConv_t.cpu().weight.data.numpy()
            np.save(path+'/w_id_t', w_spatial)
            w_trans = model.TransConv.cpu().weight.data.numpy()
            np.save(path+'/w_trans', w_trans)