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

Follow the [tools/README.md](README.md) to prepare extra caption for all regions of images. (Not necessary)

## Run
Need at least 8 GPUs 3090 to run.

Run the training without synthetic captions. 

    cd run_scripts/pmr
    bash train_pmr_huge.sh
    bash evaluate_pmr_huge.sh

Run the training with synthetic captions. (improve ~0.6 acc in test set. The final submission.)

    cd run_scripts/pmr
    bash train_pmr_huge_syn_caption.sh
    bash evaluate_pmr_huge_syn_caption.sh

Run the training on train set and val set: (Should improve 0.7 acc in test set. Haven't submit.)

    python split_val.py
    cd run_scripts/pmr
    bash train_pmr_huge_syn_caption_trainval_new.sh
    bash evaluate_pmr_huge_syn_caption_trainval_new.sh

Ensemble several models' prediction: (improve 0.8 acc in test set.)

    python ensemble_vote.py
