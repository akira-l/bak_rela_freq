import os, sys
import numpy as np

import torch

import pdb

if __name__ == '__main__':
    data_dict = torch.load('./save_dict150-20.pt')
    rela_data = data_dict['relationships']
    obj_ind = data_dict['ind_to_classes']
    rela_ind = data_dict['ind_to_predicates']
    cls_list = data_dict['gt_classes']
    obj_num = 84
    obj_sub = []
    for rela_counter in range(len(rela_data)):
        rela = rela_data[rela_counter]
        cls_ = cls_list[rela_counter]
        rela_select = np.where(cls_[rela[:, 1]] == obj_num)[0]
        if rela_select.size == 0:
            continue
        else:
            for pos in rela_select:
                try:
                    obj_sub.append([cls_[rela[pos, 0]], rela[pos, 2]])
                except:
                    pdb.set_trace()

    pharse_list = [[obj_ind[x[0]], rela_ind[x[1]]] for x in obj_sub]
    rec_file = open('rec-'+obj_ind[obj_num]+'.txt', 'a')
    for rec_item in pharse_list:
        rec_file.write(','.join(rec_item)+'\n')
    rec_file.close()
        
    pdb.set_trace()

