# PMR 

## Data prepare
Prepare the PMR dataset and unzip on dataset folder.

```
OFA/
├── checkpoint/
│   ├── ofa_base.pt
│   ├── ofa_huge.pt
│   ├── caption_large_best_clean.pt
│   └── ...
├── criterions/
├── data/
├── dataset/
│   ├── images/
│   ├── train-ori.jsonl
│   ├── val-ori.jsonl
│   ├── test-ori-without-label.jsonl
│   └── ...
├── fairseq/
├── models/
├── run_scripts/
├── tasks/
├── train.py
├── trainer.py
└── utils/
```

Follow the [README.md](README.md), to prepare the installation.

Follow the [tools/README.md](README.md) to prepare extra caption for all regions of images.

## Run
Need at least 8 GPUs 3090 to run.

    cd run_scripts/pmr
    bash run_scripts/pmr/train_pmr_huge_syn_caption.sh
    
Run the training on train set and val set:

    python split_val.py
    cd run_scripts/pmr
    bash run_scripts/pmr/train_pmr_huge_syn_caption.sh
    