from collections import defaultdict

import os

import jsonlines
import numpy as np

def read_jsonl(fname) -> list:
    data = []
    with open(fname, 'r+', encoding="utf8") as f:
        for item in jsonlines.Reader(f):
            data.append(item)
    return data


def save_jsonl(fname, data):
    assert isinstance(data, list)
    with jsonlines.open(fname, mode='w') as f:
        for d in data:
            f.write(d)

if __name__ == '__main__':
    result_list = ['checkpoints_huge_syn_caption/20_2e-5/ofa_huge_bs32_20epoch_lr2e-5_val88.9_test_text_predict.jsonl',
                   'checkpoints_huge_syn_caption_trainval/20_2e-5/ofa_huge_syn_caption_trainval_predict_test.jsonl',
                   'checkpoints_huge_syn_caption_bbox_info/20_2e-5/ofa_huge_syn_caption_bbox_info_bs32_20epoch_lr2e-5_val88.8_test_text_predict.jsonl',
                   'checkpoints_huge_bs32/20_2e-5/ofa_huge_syn_caption_20epoch_lr2e-5_val88.7_test_text_predict.jsonl']
    
    id2results = defaultdict(lambda :[0] * 4)
    id2img_id = dict()
    
    all_results = []
    
    for file in result_list:
        for r in read_jsonl(os.path.join('run_scripts/pmr', file)):
            id2results[r['total_id']][r['prediction']] += 1
            id2img_id[r['total_id']] = r['img_id']
            
    voted_results = []
    
    id2results_vote = defaultdict(int)
    for k, v in id2results.items():
        voted_results += [dict(total_id=k, img_id=id2img_id[k], prediction=int(np.argmax(v)))]
        
    save_jsonl('voted_test_prediction2.jsonl', voted_results)
    
    