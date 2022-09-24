import os

import json
import yaml
from PIL import Image
from torchvision.transforms import InterpolationMode

from tqdm import tqdm
import torch
import argparse

from tools.BLIP.models.blip import blip_decoder
from tools.utils import read_json, read_jsonl

from torchvision import transforms


blip_transform_test = transforms.Compose([
    transforms.Resize((384, 384), interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])


@torch.no_grad()
def generate_one_image_captions_for_roi(
        args,
        model,
        image,
        rois,
        transform,
        expand=1.5,
    ):
    assert expand > 1
    bboxs_img = []
    
    w_image, h_image = image.size
    for (x1, y1, x2, y2, s) in rois.values():
        w_org, h_org = (x2 - x1), (y2 - y1)

        if w_org * h_org < 3000:
            w, h = (x2 - x1) * expand, (y2 - y1) * expand
            dw = w - w_org
            dh = h - h_org

            x1, y1, x2, y2 = max(0, x1 - dw / 2), max(0, y1 - dh / 2), min(x2 + dw / 2, w_image), min(y2 + dh / 2, h_image)
        else:
            pass

        box = image.crop((x1, y1, x2, y2))
        if transform:
            box = transform(box)
        
        bboxs_img.append(box)
        
    if args.add_whole_img:
        if transform:
            image_tf = transform(image)
        bboxs_img.append(image_tf)

    bboxs_img = torch.stack(bboxs_img, dim=0)
    bboxs_img = bboxs_img.cuda(non_blocking=True)

    captions = model.generate(bboxs_img,
                              sample=False,
                              num_beams=config['num_beams'],
                              max_length=config['max_length'],
                              min_length=config['min_length'],
                              repetition_penalty=1.1)
    return captions


def generate_roi_caption(args, img_db, anno_dir, transform=None, save_dir='./dataset/blip_caption'):
    anno = read_jsonl(anno_dir)
    
    #### Model ####
    print("Creating model")
    if torch.distributed.get_rank()==0:
        model = blip_decoder(pretrained=config['pretrained'], image_size=config['image_size'], vit=config['vit'],
                             prompt=config['prompt'])
        torch.distributed.barrier()
    else:
        torch.distributed.barrier()
        model = blip_decoder(pretrained=config['pretrained'], image_size=config['image_size'], vit=config['vit'],
                             prompt=config['prompt'])
    model = model.cuda()

    caption_map = dict()
    for i, data in enumerate(tqdm(anno)):
        img_fn = data['img_fn'].replace('.../', '.__/')  # invalid path
        img_fn = img_fn.replace('./', '/')
        img_dir = img_db + img_fn
        img = Image.open(img_dir)
        metadata_fn = data['metadata_fn'].replace('.../', '.__/')  # invalid path
        metadata_fn = metadata_fn.replace('./', '/')
        metadata_dir = img_db + metadata_fn
        metadata = read_json(metadata_dir)
        rois = dict(zip(metadata['names'], metadata['boxes']))
        
        captions = generate_one_image_captions_for_roi(args,
                                                       model,
                                                       img,
                                                       rois,
                                                       transform)
        caption_map[data['total_id']] = captions
        
    file_name = os.path.splitext(os.path.split(anno_dir)[1])[0]
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    save_path = os.path.join(save_dir, f"{file_name}.json")
    json.dump(caption_map, open(save_path, 'w'))


def get_args():
    parser = argparse.ArgumentParser()
    
    # blip things
    parser.add_argument('--config', default='tools/BLIP/configs/nocaps_large.yaml')
    
    # data
    parser.add_argument('--add_whole_img', action='store_false')
    parser.add_argument('--save_dir', default='./dataset/blip_caption/')
    
    # parallel inference
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--gpu_id", type=int, default=0)
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    num_gpus = int(os.getenv('WORLD_SIZE', 1))
    args.distributed = num_gpus > 1
    if args.distributed:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(args.local_rank)
    
    annos = [
        "dataset/train-ori.jsonl",
        "dataset/train-adv.jsonl",
        "dataset/val-ori.jsonl",
        "dataset/val-adv.jsonl",
        "dataset/test-ori-without-label.jsonl"
    ]
    
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    
    assert torch.distributed.get_world_size() == len(annos)
    
    # for anno in annos:
    anno = annos[torch.distributed.get_rank()]

    generate_roi_caption(args=args,
                         img_db="dataset/images/",
                         transform=blip_transform_test,
                         anno_dir=anno,
                         save_dir=args.save_dir)
