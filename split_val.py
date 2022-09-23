import os

import json

from tools.utils import read_jsonl, save_jsonl, read_json

data = read_jsonl('dataset/val-ori.jsonl')
data_predict = read_jsonl('run_scripts/pmr/checkpoints_huge_syn_caption/20_2e-5/val_text_predict.jsonl')


correct = []

un_correct = []

total_id2predict = {item['total_id']:item['prediction'] for item in data_predict}

for item in data:
    if item['answer_label'] == total_id2predict[item['total_id']]:
        correct.append(item)
    else:
        un_correct.append(item)
        
# print(correct)
print(len(correct), len(un_correct))
print(type(correct), type(un_correct))
save_jsonl( 'dataset/val-ori-easy.jsonl', correct)
save_jsonl('dataset/val-ori-hard.jsonl', un_correct)

if os.path.exists('dataset/blip_caption/train-ori.json'):
    caption_data = read_json('dataset/blip_caption/val-ori.json')
    correct_caption = {}
    un_correct_caption = {}
    for item in correct:
        correct_caption[str(item['total_id'])] = caption_data[str(item['total_id'])]
    
    for item in un_correct:
        un_correct_caption[str(item['total_id'])] = caption_data[str(item['total_id'])]
    
    json.dump(correct_caption, open('dataset/blip_caption/val-ori-easy.json', 'w'))
    json.dump(un_correct_caption, open('dataset/blip_caption/val-ori-hard.json', 'w'))
