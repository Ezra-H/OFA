import copy
import torch
from collections import OrderedDict
import os
path = 'run_scripts/pmr/checkpoints_huge_syn_caption_trainval/20_2e-5'
file_name = 'checkpoint.best_pmr_score_0.9950.pt'
d = torch.load(os.path.join(path, file_name))

new_orderdict1 = OrderedDict()
new_orderdict2 = OrderedDict()

count = 0

for k, v in d['model'].items():
    if count < 1200:
        new_orderdict1[k] = v
    else:
        new_orderdict2[k] = v
    count += 1
    
d1 = copy.deepcopy(d)
d2 = copy.deepcopy(d)

d1['model'] = new_orderdict1
d2['model'] = new_orderdict2

d1['model_index'] = 0
d2['model_index'] = 1

file_name, ext_ = os.path.splitext(file_name)
torch.save(d1, os.path.join(path, f'{file_name}_part0.{ext_}'))
torch.save(d2, os.path.join(path, f'{file_name}_part1.{ext_}'))

