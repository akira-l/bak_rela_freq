from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import argparse
import datetime
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim 
from torch.autograd import Variable 
import torch.nn.functional as F
import torchvision.utils as tushow
torch.backends.cudnn.enabled=False

import pdb


def parse_args():
    parser = argparse.ArgumentParser(description='agent main')
    parser.add_argument('--epoch', dest='epoch',
                        default=600, type=int)
    parser.add_argument('--save_prefix', dest='save_prefix',
                        default="./save", type=str)
    parser.add_argument('--bs', dest='batch_size',
                        default=256, type=int)
    parser.add_argument('--lr', type=float,
                        default=0.0007)
    parser.add_argument('--node_num', type=int,
                        default=60)
    parser.add_argument('--set_start', type=int,
                        default=0)
    parser.add_argument('--set_num', type=int,
                        default=500)
    args = parser.parse_args()
    return args




def load_data():
    data_file = './gen_data500.txt'
    data = open(data_file)
    data_list = [x[:-1] for x in data.readlines()]
    data_list = [[float(x) for x in data_piece.split(', ')] for data_piece in data_list]
    return data_list


def Linear(in_features, out_features, bias=True, dropout=0):
    """Weight-normalized Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features, bias=bias)
    m.weight.data.uniform_(-0.1, 0.1)
    if bias:
        m.bias.data.uniform_(-0.1, 0.1)
    return m

def count_p_r(cur_pred, gt):
    val, pos = cur_pred.max(1)
    precision = (pos == gt).float().sum().item() / gt.nelement()
    return precision



def cls_count_p(cur_pred, gt):
    val, pos = cur_pred.max(1)
    correct = (pos==gt).float()
    correct_list = ((correct*(gt+1).float())-1).tolist()
    gt_set = list(set(gt.tolist()))
    for gt_num in gt_set:
        if gt_num == -1:
            continue
        gt_idx = rela_index[gt_num]
        prec_tmp = correct_list.count(gt_num)/gt.tolist().count(gt_num)
        if prec_tmp > 1:
            pdb.set_trace()
        
        rela_prec_dict[gt_idx] = max(prec_tmp, rela_prec_dict[gt_idx])
        rela_appr_dict[gt_idx] += 1
        if prec_tmp > 0.1:
            rela_prec_mean_dict[gt_idx] += prec_tmp
            rela_appr_mean_dict[gt_idx] += 1


def message_record(log_path, content):
    file_obj = open(log_path, 'a')
    file_obj.write('args:'+content+'\n')
    file_obj.close() 

def write_log(log_path, record_list, name_list):
    file_obj = open(log_path, 'a')
    content = ''
    for val, name in zip(record_list, name_list):
        content += '_' + name + '=' + str(val) + '_'
    content += '\n'
    file_obj.write(content)
    file_obj.close()

def result_plot(target_path, save_path, key_word):
    file_obj = open(target_path, 'r')
    logs = file_obj.readlines()

    color_list = ['b', 'g', 'r', 'm', 'y', 'k']
    plt.cla()
    key_word_num = len(key_word)
    for key_word_counter in range(key_word_num):
        pattern = '.*'+key_word[key_word_counter]+'=(.*)'
        if key_word_counter < key_word_num -1:
            pattern += '__'
            pattern += key_word[key_word_counter+1][0]+'.*'
        else:
            pattern += '_.*'
        plot_var = [float(re.findall(pattern, x)[0]) for x in logs]
        coordx = [x for x in range(len(plot_var))]
        plt.plot(coordx, plot_var, color_list[key_word_counter])
    label = key_word
    plt.legend(label, loc='upper left')
    name = '_'.join(key_word) + '.png'
    plt.savefig(os.path.join(save_path, name))

def reset_rela(tmp_data):
    rela = [sub_data[-1] for sub_data in tmp_data]
    rela = list(set(rela))
    for data_num in range(len(tmp_data)):
        tmp_data[data_num][-1] = rela.index(tmp_data[data_num][-1])
    return tmp_data, rela

class cls_model(nn.Module): 
    def __init__(self):
        super(cls_model, self).__init__()
        self.obj_linear1 = Linear(300, 200)
        self.obj_linear2 = Linear(210, 160)

        self.sub_linear1 = Linear(300, 200)
        self.sub_linear2 = Linear(210, 160)

        self.rela_linear1 = Linear(320+10, 240)
        self.rela_linear2 = Linear(240, rela_set_num, bias=False)
        #self.linear3 = Linear(300, 300)
        self.drop = 0.3
        
    def forward(self, x, training=True):
        obj = x[:, 0, :300]
        obj_pos = x[:, 0, 300:]
        sub = x[:, 1, :300]
        sub_pos = x[:, 1, 300:]
        pos = x[:, 2, 300:]
        obj_l1 = F.dropout(self.obj_linear1(obj), p=self.drop, training=training)
        obj_l2 = F.dropout(self.obj_linear2(torch.cat([obj_l1, obj_pos], 1)), p=self.drop, training=training)

        sub_l1 = F.dropout(self.sub_linear1(sub), p=self.drop, training=training)
        sub_l2 = F.dropout(self.sub_linear2(torch.cat([sub_l1, sub_pos], 1)), p=self.drop, training=training)

        rela_in = torch.cat([obj_l2, sub_l2, pos], 1)
        rela_l1 = self.rela_linear1(rela_in)
        rela_l2 = self.rela_linear2(rela_l1)
        return rela_l2


class cls_loss(nn.Module):
    def __init__(self):
        super(cls_loss, self).__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, pred, label):
        return self.loss(pred, label.long())



def train(args, train_data):
    log_path = './train_log.txt'
    precision_path = './precision.txt'
    if os.path.exists(precision_path):
        os.remove(precision_path)
    save_dir = './save'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    precision_checkpoint = 100

    src_data = torch.tensor(train_data)
    obj_loca = torch.zeros(len(src_data), 20)
    obj_loca[:, :4] = src_data[:, 2:6]/500
    obj_loca[:, 5] = obj_loca[:, 2]-obj_loca[:, 0]
    obj_loca[:, 6] = obj_loca[:, 3]-obj_loca[:, 1]
    src_ind = src_data[:, 0].int() -1
    obj_tensor = torch.cat([obj_vec[src_ind.tolist()].float(), obj_loca], 1)
    sub_loca = torch.zeros(len(src_data), 20)
    sub_loca[:, :4] = src_data[:, 6:10]/500
    sub_loca[:, 5] = sub_loca[:, 2]-sub_loca[:, 0]
    sub_loca[:, 6] = sub_loca[:, 3]-sub_loca[:, 1]
    src_ind = src_data[:, 2].int() -1
    sub_tensor = torch.cat([obj_vec[src_ind.tolist()].float(), sub_loca], 1)
    src_ind = src_data[:, -1].int()
    rela_tensor = torch.cat([torch.zeros(src_data.size(0), 300), src_data[:, :-1]], 1)
    inputs = torch.cat([obj_tensor.unsqueeze(1), sub_tensor.unsqueeze(1), rela_tensor.unsqueeze(1)], 1)
    inputs = Variable(inputs).float().cuda()
    label = torch.tensor([x[-1] for x in train_data])
    gt = Variable(label).cuda()
    model = cls_model()
    model.cuda()
    model.train()
    loss = cls_loss()
    optimizer = optim.SGD(model.parameters(), 
                lr=1e-4, 
                momentum=0.9, 
                weight_decay=1e-4)
    if not data_total_size < 2*args.batch_size:
        bs = args.batch_size
    else:
        bs = data_total_size // 5
    
    if bs == 0:
        return 0
    
    epoch = args.epoch
    iteration = len(train_data) // bs
    count_bar = tqdm(total=epoch)
    for epoch_counter in range(epoch):
        for iter_counter in range(iteration):
            in_tmp = inputs[iter_counter*bs:(iter_counter+1)*bs]
            gt_tmp = gt[iter_counter*bs:(iter_counter+1)*bs]
            out_tmp = model(in_tmp)
            loss_tmp = loss(out_tmp, gt_tmp)

            optimizer.zero_grad()
            loss_tmp.backward()
            optimizer.step()
            
            if (iter_counter % precision_checkpoint == 0) and (iter_counter != 0):
                precision_ = count_p_r(out_tmp, gt_tmp)
                
                write_log(precision_path,
                    [epoch_counter,
                    epoch, 
                    iter_counter, 
                    iteration, 
                    round(precision_, 4)], 
                    ['current_epoch',
                        'total_epoch', 
                        'current_times',
                        'iteration',
                        'precision'])
                
        if ( epoch_counter % 10 == 0 ) and (epoch_counter != 0):           
            torch.save(model.state_dict(), save_dir+'/cls_save'+str(epoch_counter)+'.pkl')
            #print('>>>>>>>\n------save checkpoint------\n>>>>>>>>')
        count_bar.update(1)
    count_bar.close()


def test(args, test_data):
    log_path = './train_log.txt'
    precision_path = './precision.txt'
    save_dir = './save'
    file_list = os.listdir('./save')
    save_num = 1
    for name in file_list:
        if name[-4:] == '.pkl':
            if name.split('_')[0] == 'cls':
                tmp_num = int(re.findall('.*save(.*).pkl', name)[0])
                save_num = tmp_num if tmp_num > save_num else save_num
    save_pkl = 'cls_save'+ str(save_num) +'.pkl'
    load_name = os.path.join(save_dir, save_pkl)
    print("load checkpoint %s" % (load_name))


    src_data = torch.tensor(train_data)
    obj_loca = torch.zeros(len(src_data), 20)
    obj_loca[:, :4] = src_data[:, 2:6]/500
    obj_loca[:, 5] = obj_loca[:, 2]-obj_loca[:, 0]
    obj_loca[:, 6] = obj_loca[:, 3]-obj_loca[:, 1]
    src_ind = src_data[:, 0].int() -1



    obj_tensor = torch.cat([obj_vec[src_ind.tolist()].float(), obj_loca], 1)
    sub_loca = torch.zeros(len(src_data), 20)
    sub_loca[:, :4] = src_data[:, 6:10]/500
    sub_loca[:, 5] = sub_loca[:, 2]-sub_loca[:, 0]
    sub_loca[:, 6] = sub_loca[:, 3]-sub_loca[:, 1]
    src_ind = src_data[:, 1].int() -1
    sub_tensor = torch.cat([obj_vec[src_ind.tolist()].float(), sub_loca], 1)
    src_ind = src_data[:, -1].int()

    rela_tensor = torch.cat([torch.zeros(src_data.size(0), 300), src_data[:, :-1]], 1)
    
    inputs = torch.cat([obj_tensor.unsqueeze(1), sub_tensor.unsqueeze(1), rela_tensor.unsqueeze(1)], 1)
    inputs = Variable(inputs).float().cuda()

    label = torch.tensor([x[-1] for x in test_data])
    gt = Variable(label).cuda()

    model = cls_model()
    model.load_state_dict(torch.load(load_name))
    #model = cls_model()
    model.cuda()
    model.eval()
    loss = cls_loss()

    if not data_total_size < 2*args.batch_size:
        bs = args.batch_size
    else:
        bs = data_total_size // 5
    
    if bs == 0:
        return 0

    iteration = len(test_data) // bs
    prec_rec = []
    for iter_counter in range(iteration):
        in_tmp = inputs[iter_counter*bs:(iter_counter+1)*bs]
        gt_tmp = gt[iter_counter*bs:(iter_counter+1)*bs]
        out_tmp = model(in_tmp, training=False)

        loss_tmp = loss(out_tmp, gt_tmp)

        # 1.correct rela in total test number
        # 2.output negetive pair 
        
        precision_ = cls_count_p(out_tmp, gt_tmp)

    
    

def recheck_rela():
        gt_prec_dict = rela_prec_dict
        gt_appr_dict = rela_appr_dict
        rela_vob_file = open('./../1600-400-500/relations_vocab.txt', 'r')
        rela_vob = rela_vob_file.readlines()
        rela = [rela_sub[:-1] for rela_sub in rela_vob]
        recheck_file = open(os.path.join(dict_save_folder, 'recheck_rela.txt'), 'a')
        reserve_file = open(os.path.join(dict_save_folder, 'reserve_rela.txt'), 'a')
        for rela_num in range(500):
            appr_weight = -1 if gt_appr_dict[rela_num]==0 else 0
            content = rela[rela_num]+' : '+str(appr_weight+gt_prec_dict[rela_num])+'\n'
            recheck_file.write(content)
            if gt_prec_dict[rela_num] >= 0 and gt_prec_dict[rela_num] < 0.5 and appr_weight!=-1:
                reserve_file.write(rela[rela_num]+' : '+str(gt_prec_dict[rela_num])+'\n')
        reserve_file.close()
        recheck_file.close()


if __name__ == '__main__':

    args = parse_args()
    node_num = args.node_num
    rela_set_start = args.set_start
    rela_set_num = args.set_num

    obj_vec = torch.load('./1600-500/obj_vec.pt')
    obj_vec = torch.cat(obj_vec, 0)

    rela_prec_dict = {}
    rela_prec_mean_dict = {}
    rela_appr_mean_dict = {}
    rela_appr_dict = {}
    for make_dict in range(rela_set_num+1):
        rela_prec_dict[make_dict] = 0
        rela_appr_dict[make_dict] = 0
        rela_prec_mean_dict[make_dict] = 0
        rela_appr_mean_dict[make_dict] = 0
    dict_save_folder = './save_dict'
    if not os.path.exists(dict_save_folder):
        os.mkdir(dict_save_folder)
    
    rela_set_start = 0

    print('\n\n\n---- start N.%d ----' %rela_set_start)
    print('>>>> load data ....')
    data = load_data()
    print('>>>> reset rela ....')
    data, rela_index = reset_rela(data)
    data_total_size = len(data)
    print('>>>> data set size %d ' % data_total_size)
    test_split = data_total_size//10
    train_data = data[:-test_split]
    test_data = data[-test_split:]
    print('>>>> start training ')
    train(args, train_data)
    print('>>>> start test')
    test(args, test_data)

    pass_counter = 0
    for rela_num in range(501):
        if rela_prec_dict[rela_num] > 0:
            pass_counter += 1
    print(rela_prec_dict)
    print('kill %d / 500' % pass_counter)

    fin_reserve_dict = {}
    record_file = './fin_record.txt'
    if os.path.exists(record_file):
        os.remove(record_file)
    fin_record = open(record_file, 'a')
    for rela_num in range(rela_set_num+1):
        if rela_appr_dict[rela_num] == 0:
            fin_reserve_dict[rela_num] = -1
        else:
            fin_reserve_dict[rela_num] = rela_prec_dict[rela_num] / rela_appr_dict[rela_num]
        fin_record.write(str(rela_num)+'  :  '+str(fin_reserve_dict[rela_num])+'\n')
    fin_record.close()
    pdb.set_trace()