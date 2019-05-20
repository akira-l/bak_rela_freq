import os, sys
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch

import pdb




class generate_data(object):
    def __init__(self):
        self.img_rela_path = './gen_data500.txt'

    def gen_pair(self):
        data_pkl = open('./1600-500/roidb.pkl', 'rb')
        roidb = pickle.load(data_pkl, encoding='bytes')

        gt_cls = []
        gt_rela = []
        gt_box = []
        for sub_roidb in roidb:
            gt_cls.append(sub_roidb[b'gt_classes'])
            gt_box.append(sub_roidb[b'boxes'])
            gt_rela.append(sub_roidb[b'gt_relations'])
        
        save_file = open(self.img_rela_path, 'a')
        count_bar = tqdm(total=len(gt_cls))
        for cls_, relas, boxes in zip(gt_cls, gt_rela, gt_box):
            cls1 = np.expand_dims(np.float32(cls_[relas[:, 0]]), axis=1)
            cls2 = np.expand_dims(np.float32(cls_[relas[:, 2]]), axis=1)
            rela = relas[:, 2]
            box1 = np.int32(boxes[relas[:, 0]])
            box2 = np.int32(boxes[relas[:, 2]])

            bsize1 = box1[:, [2,3]] - box1[:, [0,1]]
            bsize2 = box2[:, [2,3]] - box2[:, [0,1]]
            bsize1 = bsize1 + np.float32(bsize1==0)*1e-3
            bsize2 = bsize2 + np.float32(bsize2==0)*1e-3
            cbox1 = (box1[:, [0,1]] + box1[:, [2,3]])/2
            cbox2 = (box2[:, [0,1]] + box2[:, [2,3]])/2
            posc1 = np.expand_dims((cbox1[:, 0] - cbox2[:, 0])/bsize2[:, 0], axis=1)
            posc2 = np.expand_dims((cbox1[:, 1] - cbox2[:, 1])/bsize2[:, 1], axis=1)
            posc3 = posc1**2
            posc4 = posc2**2
            posc5 = np.expand_dims(np.log(bsize1[:, 0]/bsize2[:, 0]), axis=1)
            posc6 = np.expand_dims(np.log(bsize1[:, 1]/bsize2[:, 1]), axis=1)
            cat_data = np.concatenate((cls1, cls2, box1, box2, bsize1, bsize2, posc1, posc2, 
                                    posc3, posc4, posc5, posc6), axis=1)
            pos_data = cat_data.tolist()
            rela_label = rela.tolist()
            for d,r in zip(pos_data,rela_label):
                save_data = str(d+[r])[1:-1]+'\n'
                save_file.write(save_data)
            count_bar.update(1)
        save_file.close()
        count_bar.close()

if __name__ == '__main__':
    gen_data = generate_data()
    gen_data.gen_pair()

