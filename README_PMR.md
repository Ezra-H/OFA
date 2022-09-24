# PMR 

## Data prepare
Prepare the PMR dataset and unzip on dataset folder.

```
OFA/
├── checkpoint/
│   ├── ofa_huge.pt
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

Follow the [tools/README.md](tools/README.md) to prepare extra caption for all regions of images. (Not necessary)

Download the ofa-huge checkpoint.

    mkdir checkpoints
    wget https://ofa-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/ofa_huge.pt

## Run
Need at least 8 GPUs 3090 to run.

Run the training without synthetic captions. (Baseline)

    cd run_scripts/pmr
    bash train_pmr_huge.sh
    bash evaluate_pmr_huge.sh

Run the training with synthetic captions. (improve ~0.6 acc in test set. The final submission.)

    cd run_scripts/pmr
    bash train_pmr_huge_syn_caption.sh
    bash evaluate_pmr_huge_syn_caption.sh

Run the training on train set and val set: (Should improve 0.7 acc in test set. Haven't submit.)

Need to depend on a trained model to evaluate on validation set. (Modify the split to 'val' in evaluate_pmr_huge_syn_caption.sh) 
Split the validation dataset to two set. 
One set is predicted correctly which added to training set. The other is predicted wrong which still be a validation set. 

    python split_val.py  
    cd run_scripts/pmr
    bash train_pmr_huge_syn_caption_trainval_new.sh
    bash evaluate_pmr_huge_syn_caption_trainval_new.sh

Ensemble several models' prediction: (improve 0.8 acc in test set. (Three models.))

    python ensemble_vote.py


