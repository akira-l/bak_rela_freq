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
import random

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
                        default="save", type=str)
    parser.add_argument('--bs', dest='batch_size',
                        default=256, type=int)
    parser.add_argument('--lr', type=float,
                        default=1e-4)
    parser.add_argument('--node_num', type=int,
                        default=60)
    parser.add_argument('--set_start', type=int,
                        default=0)
    parser.add_argument('--set_num', type=int,
                        default=179)
    parser.add_argument('--shuffle', dest='shuffle', action='store_true')
    parser.add_argument('--data_dir', dest='data_dir', default='./save-centered180', type=str)
    args = parser.parse_args()
    return args

#load graph
def load_glove(glove_path):
    print('loading glove .... ')
    fp = open(glove_path, 'r', encoding='utf-8')
    vectors = {}
    for line in fp:
        vals = line.rstrip().split(' ')
        vectors[vals[0]] = np.zeros((1, 300))
        vectors[vals[0]][0, :] = list(map(float, vals[1:]))
    print('load glove over !')
    return vectors


def load_data():
    data_file = './gen_data500-centered180.txt'
    data = open(data_file)
    data_list = [x[2:-1] for x in data.readlines()]
    obj_name = [x[0] for x in data.readlines()]
    sub_name = [x[1] for x in data.readlines()]
    data_list = [[float(x) for x in data_piece.split(', ')] for data_piece in data_list]
    return obj_name, sub_name, data_list


def get_wordvec(word):
    if word in glove:
        return torch.tensor(glove[word])
    else:
        word_split = word.split(' ')
        for sub_word in word_split:
            if not sub_word in glove:
                print('word %s not in glove vector ???' % word)
                pdb.set_trace()
        vec_tmp = glove[word_split[0]]
        for sub_word in range(1, len(word_split)):
            vec_tmp += torch.tensor(glove[glove[sub_word]])
        return vec_tmp / len(word)


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
        gt_idx = gt_num
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
        self.vec_size = 300
        self.obj_linear1 = Linear(self.vec_size, 60)
        self.obj_linear2 = Linear(70, 240)

        self.sub_linear1 = Linear(self.vec_size, 60)
        self.sub_linear2 = Linear(70, 240)

        self.rela_linear1 = Linear(480+22, 400)
        self.rela_linear2 = Linear(400, rela_set_num, bias=False)
        #self.linear3 = Linear(300, 300)
        self.drop = 0.001
        self.bn_norm = nn.BatchNorm1d(480+22, momentum=0.5)

    def forward(self, x, training=True):
        obj = x[:, 0, :self.vec_size]
        obj_pos = x[:, 0, self.vec_size:]
        sub = x[:, 1, :self.vec_size]
        sub_pos = x[:, 1, self.vec_size:]
        pos = x[:, 2, -22:]
        obj_l1 = F.dropout(self.obj_linear1(obj), p=self.drop, training=training)
        obj_l2 = F.dropout(self.obj_linear2(torch.cat([obj_l1, obj_pos], 1)), p=self.drop, training=training)

        sub_l1 = F.dropout(self.sub_linear1(sub), p=self.drop, training=training)
        sub_l2 = F.dropout(self.sub_linear2(torch.cat([sub_l1, sub_pos], 1)), p=self.drop, training=training)

        rela_in = torch.cat([obj_l2, sub_l2, pos], 1)
        rela_in = self.bn_norm(rela_in)
        rela_l1 = self.rela_linear1(rela_in)
        rela_l2 = self.rela_linear2(rela_l1)
        return rela_l2


class cls_loss(nn.Module):
    def __init__(self):
        super(cls_loss, self).__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, pred, label):
        return self.loss(pred, label.long())



def train(args, train_data_gather):
    train_obj_name, train_sub_name, train_data = train_data_gather
    precision_path = './'+args.save_prefix+'precision.txt'
    if os.path.exists(precision_path):
        os.remove(precision_path)
    save_dir = args.data_dir
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    precision_checkpoint = 100


    label = torch.tensor([x[-1] for x in train_data])
    gt = Variable(label).cuda()

    model = cls_model()
    model.cuda()
    model.train()
    loss = cls_loss()
    optimizer = optim.SGD(model.parameters(),
                lr=args.lr,#1e-4,
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
    res_data = len(train_data) % bs
    count_bar = tqdm(total=epoch)
    total_len = len(train_data)
    shuffle_idx = list(range(total_len))
    for epoch_counter in range(epoch):
        if args.shuffle:
            random.shuffle(shuffle_idx)
        for iter_counter in range(iteration):
            if args.shuffle:
                given_idx = torch.tensor(shuffle_idx)
                train_batch_tmp = train_data[given_idx[iter_counter*bs:(iter_counter+1)*bs]]
                train_obj_tmp = train_obj_name[given_idx[iter_counter*bs:(iter_counter+1)*bs]]
                train_sub_tmp = train_sub_name[given_idx[iter_counter*bs:(iter_counter+1)*bs]]
                gt_tmp = gt[given_idx[iter_counter*bs:(iter_counter+1)*bs]]
            else:
                train_batch_tmp = train_data[given_idx[iter_counter*bs:(iter_counter+1)*bs]]
                train_obj_tmp = train_obj_name[given_idx[iter_counter*bs:(iter_counter+1)*bs]]
                train_sub_tmp = train_sub_name[given_idx[iter_counter*bs:(iter_counter+1)*bs]]
                gt_tmp = gt[iter_counter*bs:(iter_counter+1)*bs]

            train_batch_tmp = torch.tensor(train_batch_tmp)
            obj_tensor = []
            sub_tensor = []
            for obj_name_tmp, sub_name_tmp in zip(train_obj_tmp, train_sub_tmp):
                obj_tensor.append(get_wordvec(obj_name_tmp))
                sub_tensor.append(get_wordvec(sub_name_tmp))
            obj_tensor = torch.cat(obj_tensor, 0)
            sub_tensor = torch.cat(sub_tensor, 0)

            obj_loca = torch.zeros(bs, 10)
            obj_loca[:, :4] = train_batch_tmp[:, :4]
            obj_loca[:, 5] = obj_loca[:, 2]-obj_loca[:, 0]
            obj_loca[:, 6] = obj_loca[:, 3]-obj_loca[:, 1]
            sub_loca = torch.zeros(bs, 10)
            sub_loca[:, :4] = train_batch_tmp[:, :4]
            sub_loca[:, 5] = sub_loca[:, 2]-sub_loca[:, 0]
            sub_loca[:, 6] = sub_loca[:, 3]-sub_loca[:, 1]

            addition_tensor = torch.zeros(train_batch_tmp.size(0), 296)
            addition_tensor[:, -4:] = (obj_loca - sub_loca)[:, :4]
            addition_tensor[:, -8:-4] = sub_loca[:, :4]
            addition_tensor[:, -12:-8] = obj_loca[:, :4]
            rela_tensor = torch.cat([addition_tensor, train_batch_tmp[:, 7:-1]], 1)
            inputs = torch.cat([obj_tensor.unsqueeze(1), sub_tensor.unsqueeze(1), rela_tensor.unsqueeze(1)], 1)
            inputs = Variable(inputs).float().cuda()

            out_tmp = model(inputs)
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
        if res_data > 0:
            in_tmp = inputs[-res_data:]
            gt_tmp = gt[-res_data:]
            out_tmp = model(in_tmp)
            loss_tmp = loss(out_tmp, gt_tmp)

            optimizer.zero_grad()
            loss_tmp.backward()
            optimizer.step()

            write_log(precision_path,
                    [epoch_counter,
                    epoch,
                    iter_counter+1,
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




def test(args, test_data_gather):
    test_obj_name, test_sub_name, test_data = test_data_gather
    save_dir = args.data_dir
    file_list = os.listdir(save_dir)
    save_num = 1
    for name in file_list:
        if name[-4:] == '.pkl':
            if name.split('_')[0] == 'cls':
                tmp_num = int(re.findall('.*save(.*).pkl', name)[0])
                save_num = tmp_num if tmp_num > save_num else save_num
    save_pkl = 'cls_save'+ str(save_num) +'.pkl'
    load_name = os.path.join(save_dir, save_pkl)
    print("load checkpoint %s" % (load_name))

    label = torch.tensor([x[-1] for x in test_data])
    gt = Variable(label).cuda()

    model = cls_model()
    model.load_state_dict(torch.load(load_name))
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
    res_data = len(test_data) % bs
    prec_rec = []
    for iter_counter in range(iteration):

        test_batch_tmp = test_data[iter_counter*bs:(iter_counter+1)*bs]
        test_obj_tmp = test_obj_name[iter_counter*bs:(iter_counter+1)*bs]
        test_sub_tmp = test_sub_name[iter_counter*bs:(iter_counter+1)*bs]
        gt_tmp = gt[iter_counter*bs:(iter_counter+1)*bs]

        test_batch_tmp = torch.tensor(test_batch_tmp)
        obj_tensor = []
        sub_tensor = []
        for obj_name_tmp, sub_name_tmp in zip(test_obj_tmp, test_sub_tmp):
            obj_tensor.append(get_wordvec(obj_name_tmp))
            sub_tensor.append(get_wordvec(sub_name_tmp))
        obj_tensor = torch.cat(obj_tensor, 0)
        sub_tensor = torch.cat(sub_tensor, 0)

        obj_loca = torch.zeros(bs, 10)
        obj_loca[:, :4] = test_batch_tmp[:, :4]
        obj_loca[:, 5] = obj_loca[:, 2]-obj_loca[:, 0]
        obj_loca[:, 6] = obj_loca[:, 3]-obj_loca[:, 1]
        sub_loca = torch.zeros(bs, 10)
        sub_loca[:, :4] = test_batch_tmp[:, :4]
        sub_loca[:, 5] = sub_loca[:, 2]-sub_loca[:, 0]
        sub_loca[:, 6] = sub_loca[:, 3]-sub_loca[:, 1]

        addition_tensor = torch.zeros(test_batch_tmp.size(0), 296)
        addition_tensor[:, -4:] = (obj_loca - sub_loca)[:, :4]
        addition_tensor[:, -8:-4] = sub_loca[:, :4]
        addition_tensor[:, -12:-8] = obj_loca[:, :4]
        rela_tensor = torch.cat([addition_tensor, test_batch_tmp[:, 7:-1]], 1)
        inputs = torch.cat([obj_tensor.unsqueeze(1), sub_tensor.unsqueeze(1), rela_tensor.unsqueeze(1)], 1)
        inputs = Variable(inputs).float().cuda()

        out_tmp = model(inputs)
        loss_tmp = loss(out_tmp, gt_tmp)

        # 1.correct rela in total test number
        # 2.output negetive pair

        cls_count_p(out_tmp, gt_tmp)

    if res_data > 0:
        in_tmp = inputs[-res_data:]
        gt_tmp = gt[-res_data:]
        out_tmp = model(in_tmp, training=False)

        loss_tmp = loss(out_tmp, gt_tmp)
        cls_count_p(out_tmp, gt_tmp)




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


    glove = load_glove('./../glove/glove.840B.300d.txt')

    vocab_file = open('./25.txt')
    fixed_rela_vocab = vocab_file.readlines()
    vocab_name = fixed_rela_vocab[2:]

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
    obj_name_gather, sub_name_gather, data_gather = load_data()
    print('>>>> reset rela ....')
    data_total_size = len(data_gather)
    print('>>>> data set size %d ' % data_total_size)
    rela_gather = [x[-1] for x in data_gather]
    gather_dict = {}
    for tmp_idx in range(rela_set_num):
        gather_dict[tmp_idx] = []
    for gather_idx in range(len(rela_gather)):
        gather_dict[rela_gather[gather_idx]].append(gather_idx)
    # get top 5 element in dict
    test_list = []
    for test_idx in range(rela_set_num):
        test_list.extend(gather_dict[test_idx][:5])
    #test_list = [gather_dict[x][:5] for x in range(499)]
    test_data_ = [data_gather[x] for x in test_list]
    test_obj_name = [obj_name_gather[x] for x in test_list]
    test_sub_name = [sub_name_gather[x] for x in test_list]
    test_list_set = set(test_list)
    train_list = list(range(len(rela_gather)))
    train_list = list(filter(lambda x: x not in test_list_set, train_list))
    train_data_ = [data_gather[x] for x in train_list]
    train_obj_name = [obj_name_gather[x] for x in train_list]
    train_sub_name = [sub_name_gather[x] for x in train_list]

    #test_split = data_total_size//10
    #train_data = data[:-test_split]
    #test_data = data[-test_split:]
    print('>>>> start training ')
    train(args, [train_obj_name, train_sub_name, train_data_])
    print('>>>> start test')
    test(args, [test_obj_name, test_sub_name, test_data_])

    pass_counter = 0
    for rela_num in range(rela_set_num):
        if rela_prec_dict[rela_num] > 0:
            pass_counter += 1
    print(rela_prec_dict)
    for name_num in range(len(vocab_name)):
        print('acc: %.4f    rela_name: %s' % (rela_prec_dict[name_num], vocab_name[name_num]))
    print('kill %d / %d' % (pass_counter, rela_set_num))
    pdb.set_trace()

    '''
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
    '''









