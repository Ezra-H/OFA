# Generate Captions
We use a caption model to generate new description for each region in images. We think the region caption will help the model to identity which object/person we are talking about.

We use BLIP to generate the model. Actually, OFA can also generate the caption.

## download the checkpoint 
    
    mkdir checkpoint
    wget https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_caption.pth -O checkpoint/blip_large_pretrain.sh  

## clone the repo

    git clone https://github.com/Ezra-H/BLIP.git tools/BLIP

## run 

    PYTHONPATH="$(pwd)":$PYTHONPATH python -m torch.distributed.launch --nproc_per_node 5 tools/blip_generate_caption_for_roi.py --add_whole_img
