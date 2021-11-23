from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from model import dataset
from model import menet
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from collections import OrderedDict
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
    parser.add_argument('--num_points', type=int, default=512, help='input batch size')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers', )
    parser.add_argument('--nepoch', type=int, default=200, help='number of epochs to train for')
    parser.add_argument('--outf', type=str,default='',help='output folder')
    parser.add_argument('--model', type=str, default='', help='path of model')
    parser.add_argument('--phase', type=str, default='train_MENet')
    parser.add_argument('--numclass', type=int, default=11,help='N-Cars(2),N-MNIST(10),DVS128(11)')
    parser.add_argument('--data_path', type=str, default='',help='number of epochs to train for')
    args = parser.parse_args()

    if args.phase == 'train_MENet_single':
        train_MENet_single(args)
        return
    if args.phase == 'train_MENet':
        train_MENet(args)
        return
    if args.phase == 'test_MENet_single':
        test_MENet_single(args)
        return
    if args.phase == 'test_MENet':
        win_len = 4
        print('win len is:', win_len)
        test_MENet(args, win_len)
        return
def train_MENet_single(args):
    classifier = menet.MENet_single(num_class=args.numclass)
    classifier = nn.DataParallel(classifier).cuda()
    optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    trainLoader = DataLoader(dataset.mnist_cars(args.data_path),batch_size=args.batchSize,shuffle=True, num_workers=int(args.workers))
    if args.model != '':
        print(args.model)
        classifier.load_state_dict((torch.load(args.model)))
    num_batch = len(trainLoader)
    for epoch in range(0, args.nepoch):
        scheduler.step()
        for i, data in enumerate(trainLoader, 0):
            points1, target = data
            target = target[:, 0]
            points1 = points1.transpose(2, 1)
            points1, target = points1.cuda(), target.cuda()
            optimizer.zero_grad()
            classifier = classifier.train()
            pred1 = classifier(points1)
            loss = F.nll_loss(pred1, target)
            loss.backward()
            optimizer.step()
            pred_choice1 = pred1.data.max(1)[1]
            correct = pred_choice1.eq(target.data).cpu().sum()
            print('[%d: %d/%d] train loss: %f accuracy: %f' % (
                epoch, i, num_batch, loss.item(), correct.item() / float(args.batchSize)))
        torch.save(classifier.state_dict(), '%s/cls_model_%d.pth' % (args.outf, epoch))
def train_MENet(args):
    classifier = menet.MENet(num_class=args.numclass)
    classifier = nn.DataParallel(classifier).cuda()
    optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    trainLoader = DataLoader(
        dataset.DVS128_aswin_diff(args.data_path),
        batch_size=args.batchSize,
        shuffle=True, num_workers=int(args.workers), drop_last=True)
    num_batch = len(trainLoader)
    if args.model != '':
        print(args.model)
        classifier.load_state_dict((torch.load(args.model)))
    for epoch in range(0, args.nepoch):
        scheduler.step()
        for i, data in enumerate(trainLoader, 0):
            points1, points2, target = data

            target = target[:, 0]

            points1 = points1.transpose(2, 1)  ##B*C*N
            points2 = points2.transpose(2, 1)  ##B*winnum*C*N

            points1, points2, target = points1.cuda(), points2.cuda(), target.cuda()
            optimizer.zero_grad()
            classifier = classifier.train()
            pred1, pred2 = classifier(points1, points2)
            loss1 = F.nll_loss(pred1, target)
            loss2 = F.nll_loss(pred2, target)
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()
            pred_choice1 = pred1.data.max(1)[1]
            pred_choice2 = pred2.data.max(1)[1]
            correct1 = pred_choice1.eq(target.data).cpu().sum()
            correct2 = pred_choice2.eq(target.data).cpu().sum()
            correct = correct1 + correct2
            print('[%d: %d/%d] train loss: %f accuracy1: %f accuracy2: %f accuracy: %f' % (
                epoch, i, num_batch, loss.item(), correct1.item() / float(args.batchSize),
                correct2.item() / float(args.batchSize), correct.item() / float(args.batchSize * 2)))

        torch.save(classifier.state_dict(), '%s/cls_model_%d.pth' % (args.outf, epoch))
def test_MENet_single(args):

    testdataloader = DataLoader(dataset.mnist_cars(args.data_path), batch_size=1,shuffle=False)
    classifier = menet.MENet_single(num_class=args.numclass)
    print(args.model)
    classifier.load_state_dict(rm_module(torch.load(args.model)))
    classifier=classifier.cuda()
    total_testset=0.0
    correct=0.0
    for i,data in enumerate(testdataloader, 0):

        points, target = data
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        classifier = classifier.eval()
        pred = classifier(points)
        pred_choice = pred.data.max(1)[1]
        correct += pred_choice.eq(target.data).cpu().sum()
        total_testset+=1
    print("final accuracy {}".format(float(correct) / float(total_testset)))

def test_MENet(args,win_len):
    testdataloader = DataLoader(
        dataset.DVS128_test_diff(args.data_path), batch_size=1,
        shuffle=False)
    classifier = menet.MENet_test(num_class=args.numclass)
    print(args.model)
    classifier.load_state_dict(rm_module(torch.load(args.model)))
    classifier = classifier.cuda()
    classifier = classifier.eval()
    total_correct = 0
    total_testset = 0
    dict = {}
    dict_label = {}
    test_file = open(args.data_path, 'r')
    for line in test_file:
        label = line.split(' ')[1].replace('\n', '')
        data_name = line.split(' ')[0].split('/')[-1].split('user')[-1].split('.txt')[0] + '_' + str(int(label) - 1)
        dict[data_name] = np.zeros(args.numclass)
        dict_label[data_name] = int(label)-1
    flag = 0
    old_target = ''
    win_num = 0
    old_points = 0
    for i, data in enumerate(testdataloader, 0):

        points,points_ori, target, name = data

        if win_num % win_len == 0:
            flag = 0
            win_num = 0
        if name[0] != old_target:
            flag = 0
            old_target = name[0]
            win_num = 0
        win_num += 1
        if flag == 0:
            target = target[:, 0]
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            pred, old_feature = classifier(points, None, None, flag, None, None)
            pred_choice = pred.data.max(1)[1]
            old_points = points_ori
            flag += 1

        else:
            result_list = np.zeros(args.numclass)
            point_set_2 = difference_cul(old_points.cpu().detach().numpy()[0], points_ori.cpu().detach().numpy()[0])
            if (len(point_set_2)) > args.num_points:
                choice = np.random.choice(len(point_set_2), args.num_points, replace=True)
                point_set_2_expand = point_set_2[choice, :]
            else:
                point_set_2_expand = np.zeros((args.num_points, 4))
                choice = np.random.choice(len(point_set_2), args.num_points - len(point_set_2), replace=True)
                point_set_2_expand[0:len(point_set_2), :] = point_set_2
                point_set_2_expand[len(point_set_2):args.num_points, :] = point_set_2[choice, :]
            point_set_min = np.min(point_set_2_expand, axis=0)
            point_set_max = np.max(point_set_2_expand, axis=0)
            range_ = point_set_max - point_set_min
            points1 = np.zeros_like(old_points.cpu().detach().numpy())
            points1[0] = (point_set_2_expand - point_set_min) / range_
            points1, target = torch.from_numpy(points1).cuda().float(), target.cuda()
            points1 = points1.transpose(2, 1)
            pred, old_feature = classifier(None, points1, old_feature, flag, None, None)
            pred_choice = pred.data.max(1)[1]
            result_list[pred_choice] += 1
        dict[name[0]][pred_choice] += 1
    all_label = dict.keys()
    for k in all_label:
        dict_name = k
        if np.where(dict[dict_name] == np.max(dict[dict_name]))[0][0] == dict_label[dict_name]:
            total_correct += 1
        total_testset += 1
    print("final accuracy {}".format(total_correct / float(total_testset)))


def rm_module(old_dict):
    new_state_dict = OrderedDict()
    for k, v in old_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:]  # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    return new_state_dict
def cal_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''
    gold = gold.contiguous().view(-1)
    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss
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



main()

#cul_flop_para()