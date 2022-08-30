#!/usr/bin/env python3 -u
import argparse
from src.data import read_jsonl

from fairseq import options
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from omegaconf import DictConfig


def main(cfg: DictConfig, **kwargs):
    data = read_jsonl(cfg.file)
    anno = read_jsonl('/data5/huangrunhui/proj7/pmr_hcp/dataset/val-ori.jsonl')

    total_id2answer_label = dict()
    for item in anno:
        total_id2answer_label[item['total_id']] = item['answer_label']

    count = 0
    correct = 0
    
    for d in data:
        count += 1
        correct += int(total_id2answer_label[d['total_id']] == d['prediction'])
    
    print(f'count: {count}, correct: {correct}, acc: {correct / count}')


def cli_main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("file", default=None, type=str, help="the val file to calculate the accuracy.")
    args = parser.parse_args()
    
    main(args)
    

if __name__ == "__main__":
    cli_main()
