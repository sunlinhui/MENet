from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import sys
from tqdm import tqdm 
import json
import pdb

def difference_cul(event1,event2):
    image_w = 128
    image_h = 128
    image_2c_1 = np.zeros((image_w, image_h, 2))
    image_2c_2 = np.zeros((image_w, image_h, 2))
    image_2c_2_t = np.zeros((image_w, image_h, 2))
    event1 = event1.astype(int)
    event2 = event2.astype(int)
    for i in range(event1.shape[0]):
        if event1[i, 2] == 1:
            image_2c_1[event1[i, 0], event1[i, 1], 0] += 1
        else:
            image_2c_1[event1[i, 0], event1[i, 1], 1] += 1
    for i in range(event2.shape[0]):
        if event2[i, 2] == 1:
            image_2c_2[event2[i, 0], event2[i, 1], 0] += 1
            image_2c_2_t[event2[i, 0], event2[i, 1], 0] = event2[i, 3]
        else:
            image_2c_2[event2[i, 0], event2[i, 1], 1] += 1
            image_2c_2_t[event2[i, 0], event2[i, 1], 1] = event2[i, 3]
    difference = image_2c_2 - image_2c_1
    d_f_p1 = np.nonzero(difference[:, :, 0])

    d_f_p2 = np.nonzero(difference[:, :, 1])
    result = np.zeros((d_f_p1[0].shape[0] + d_f_p2[0].shape[0], 4))
    result[0:d_f_p1[0].shape[0], 3] = image_2c_2_t[d_f_p1[0], d_f_p1[1], 0]
    result[0:d_f_p1[0].shape[0], 0] = d_f_p1[0]
    result[0:d_f_p1[0].shape[0], 1] = d_f_p1[1]
    result[0:d_f_p1[0].shape[0], 2] = difference[d_f_p1[0], d_f_p1[1], 0]

    result[d_f_p1[0].shape[0]:d_f_p1[0].shape[0] + d_f_p2[0].shape[0], 3] = image_2c_2_t[d_f_p2[0], d_f_p2[1], 1]
    result[d_f_p1[0].shape[0]:d_f_p1[0].shape[0] + d_f_p2[0].shape[0], 0] = d_f_p2[0]
    result[d_f_p1[0].shape[0]:d_f_p1[0].shape[0] + d_f_p2[0].shape[0], 1] = d_f_p2[1]
    result[d_f_p1[0].shape[0]:d_f_p1[0].shape[0] + d_f_p2[0].shape[0], 2] = -difference[d_f_p2[0], d_f_p2[1], 1]
    return result

class mnist_cars(data.Dataset):
    def __init__(self,
                 root,
                 npoints=512 # N-Cars(1024) N-MINIST(512)
                 ):
        self.npoints = npoints
        self.root = root
        self.file = open(root,'r')
        data=''
        label=''
        for line in self.file:
            data=data+line.split(' ')[0]+' '
            label=label+line.split(' ')[1].replace('\n','')+' '
        self.all_data=data.split(' ')
        self.all_label=label.split(' ')
    def __getitem__(self, index):
        data_name=self.all_data[index]
        label_index=int(self.all_label[index])
        event_data=np.loadtxt(data_name)
        choice = np.random.choice(len(event_data), self.npoints, replace=True)
        point_set = event_data[choice, :]
        point_set[:, 2]=(point_set[:,2]+1)/2
        t=point_set.copy()
        point_set_min=np.min(point_set,axis=0)
        point_set_max=np.max(point_set,axis=0)
        range=point_set_max-point_set_min
        point_set2=(point_set-point_set_min)/range
        point_set2[:,2]=t[:,2]
        data_label=label_index
        point_set2 = torch.from_numpy(point_set2.astype(np.float32))
        cls = torch.from_numpy(np.array([data_label]).astype(np.int64))
        return point_set2, cls
    def __len__(self):
        return len(self.all_data)-1

class DVS128_aswin_diff(data.Dataset):
    def __init__(self,
                 root,
                 npoints=512
                 ):
        self.npoints = npoints
        self.root = root
        self.file = open(root,'r')
        data=''
        label=''
        for line in self.file:
            data=data+line.split(' ')[0]+' '
            label=label+line.split(' ')[1].replace('\n','')+' '
        self.all_data=data.split(' ')
        self.all_label=label.split(' ')
    def __getitem__(self, index):
        next_index = index + 1
        data_name1 = self.all_data[index]
        data_name2 = self.all_data[next_index]
        if next_index >= 29516:# the number of test samples
            next_index = index - 1
            data_name2 = self.all_data[next_index]
        data_index_name1 = data_name1.split('/')[-1].split('user')[-1].split('.txt')[0]
        data_index_name2 = data_name2.split('/')[-1].split('user')[-1].split('.txt')[0]
        if data_index_name1 != data_index_name2:
            next_index = index - 1
            data_name2 = self.all_data[next_index]

        label_index = int(self.all_label[index])
        event_data = np.loadtxt(data_name1)
        choice = np.random.choice(len(event_data), self.npoints, replace=True)
        point_set = event_data[choice, :]
        t = point_set.copy()
        point_set_min = np.min(point_set, axis=0)
        point_set_max = np.max(point_set, axis=0)
        range = point_set_max - point_set_min
        point_set1 = (point_set - point_set_min) / range
        point_set1[:, 2] = t[:, 2]

        event_data = np.loadtxt(data_name2)
        choice = np.random.choice(len(event_data), self.npoints, replace=True)
        point_set_2 = event_data[choice, :]
        point_set_2 = difference_cul(point_set, point_set_2)
        if (len(point_set_2))>self.npoints:
            choice = np.random.choice(len(point_set_2), self.npoints, replace=True)
            point_set_2 = point_set_2[choice, :]
        else:
            point_set_2_expand=np.zeros((self.npoints,4))
            choice = np.random.choice(len(point_set_2), self.npoints-len(point_set_2), replace=True)
            point_set_2_expand[0:len(point_set_2),:]=point_set_2
            point_set_2_expand[len(point_set_2):self.npoints,:]=point_set_2[choice,:]
            point_set_2=point_set_2_expand

        point_set_min = np.min(point_set_2, axis=0)
        point_set_max = np.max(point_set_2, axis=0)
        range = point_set_max - point_set_min
        point_set2 = (point_set_2 - point_set_min) / range
        data_label = label_index - 1
        point_set1 = torch.from_numpy(point_set1.astype(np.float32))
        point_set2 = torch.from_numpy(point_set2.astype(np.float32))
        cls = torch.from_numpy(np.array([data_label]).astype(np.int64))
        return point_set1,point_set2, cls

    def __len__(self):
        return len(self.all_data)-1
class DVS128_test_diff(data.Dataset):
    def __init__(self,
                 root,
                 npoints=512
                 ):
        self.npoints = npoints
        self.root = root
        self.file = open(root,'r')
        data=''
        label=''
        for line in self.file:
            data=data+line.split(' ')[0]+' '
            label=label+line.split(' ')[1].replace('\n','')+' '
        self.all_data=data.split(' ')
        self.all_label=label.split(' ')



    def __getitem__(self, index):
        data_name=self.all_data[index]
        data_index_name=data_name.split('/')[-1].split('user')[-1].split('.txt')[0]+'_'+str(int(self.all_label[index])-1)
        label_index=int(self.all_label[index])
        event_data=np.loadtxt(data_name)
        choice = np.random.choice(len(event_data), self.npoints, replace=True)
        point_set = event_data[choice, :]
        t = point_set.copy()
        point_set_min = np.min(point_set, axis=0)
        point_set_max = np.max(point_set, axis=0)
        range = point_set_max - point_set_min
        point_set2 = (point_set - point_set_min) / range
        point_set2[:, 2] = t[:, 2]
        data_label=label_index-1
        point_set2 = torch.from_numpy(point_set2.astype(np.float32))
        cls = torch.from_numpy(np.array([data_label]).astype(np.int64))
        return point_set2, point_set,cls,data_index_name

    def __len__(self):
        return len(self.all_data)-1


