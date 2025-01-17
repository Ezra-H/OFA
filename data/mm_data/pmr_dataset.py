# Copyright 2022 The OFA-Sys Team. 
# All rights reserved.
# This source code is licensed under the Apache 2.0 license 
# found in the LICENSE file in the root directory.

from io import BytesIO

import logging
import warnings

import numpy as np
import torch
import base64
from torchvision import transforms
import utils.transforms as T

from PIL import Image, ImageFile

from data import data_utils
from data.ofa_dataset import OFADataset

ImageFile.LOAD_TRUNCATED_IMAGES = True
ImageFile.MAX_IMAGE_PIXELS = None
Image.MAX_IMAGE_PIXELS = None

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def collate(samples, pad_idx, eos_idx):
    if len(samples) == 0:
        return {}

    def merge(key):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx,
            eos_idx=eos_idx,
        )

    id = np.array([s["id"] for s in samples])
    img_id = np.array([s["img_id"] for s in samples])
    ans_pos_idx = np.array([s["ans_pos_idx"] for s in samples])
    answer_label = np.array([s["answer_label"] for s in samples])

    src_tokens = merge("source")
    src_lengths = torch.LongTensor([s["source"].ne(pad_idx).long().sum() for s in samples])

    patch_images = torch.stack([sample['patch_image'] for sample in samples], dim=0)
    patch_masks = torch.cat([sample['patch_mask'] for sample in samples])

    ref_dict = None
    if samples[0].get("ref_dict", None) is not None:
        ref_dict = np.array([s['ref_dict'] for s in samples])

    constraint_masks = None
    if samples[0].get("constraint_mask", None) is not None:
        constraint_masks = merge("constraint_mask")

    decoder_prompts = None
    if samples[0].get("decoder_prompt", None) is not None:
        decoder_prompts = np.array([s['decoder_prompt'].tolist() for s in samples])

    prev_output_tokens = None
    target = None
    if samples[0].get("target", None) is not None:
        target = merge("target")
        tgt_lengths = torch.LongTensor(
            [s["target"].ne(pad_idx).long().sum() for s in samples]
        )
        ntokens = tgt_lengths.sum().item()

        if samples[0].get("prev_output_tokens", None) is not None:
            prev_output_tokens = merge("prev_output_tokens")
    else:
        ntokens = src_lengths.sum().item()

    batch = {
        "id": id,
        "nsentences": len(samples),
        "ntokens": ntokens,
        "net_input": {
            "src_tokens": src_tokens,
            "src_lengths": src_lengths,
            "patch_images": patch_images,
            "patch_masks": patch_masks,
            "prev_output_tokens": prev_output_tokens
        },
        "ref_dict": ref_dict,
        "constraint_masks": constraint_masks,
        "decoder_prompts": decoder_prompts,
        "target": target,
        "img_id": img_id,
        "ans_pos_idx": ans_pos_idx,
        "answer_label": answer_label,
    }

    return batch


class PMRDataset(OFADataset):
    def __init__(
        self,
        split,
        dataset,
        bpe,
        src_dict,
        tgt_dict=None,
        max_src_length=80,
        max_tgt_length=30,
        patch_image_size=224,
        add_caption=False,
        constraint_trie=None,
        imagenet_default_mean_and_std=False,
        prompt_type="none"
    ):
        super().__init__(split, dataset, bpe, src_dict, tgt_dict)
        self.max_src_length = max_src_length
        self.max_tgt_length = max_tgt_length
        self.patch_image_size = patch_image_size

        self.add_caption = add_caption
        self.constraint_trie = constraint_trie
        self.prompt_type = prompt_type

        if imagenet_default_mean_and_std:
            mean = IMAGENET_DEFAULT_MEAN
            std = IMAGENET_DEFAULT_STD
        else:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]

        self.patch_resize_transform = transforms.Compose([
            lambda image: image.convert("RGB"),
            transforms.Resize((patch_image_size, patch_image_size), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        
        self.patch_resize_transform = T.Compose([
            lambda image, target: (image.convert("RGB"), target),
            T.Resize((patch_image_size, patch_image_size)),
            # T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std, max_image_size=patch_image_size)
        ])
        
        self.num_bins = 1000

    def __getitem__(self, index):
        # uniq_id, image, hypothesis, caption, label = self.dataset[index]
        data_frame = self.dataset[index]

        uniq_id = data_frame['total_id']
        img_id = data_frame['img_id']
        image = data_frame['img_dir']
        hypothesis = data_frame['premise']
        caption = data_frame['answer']
        label = data_frame['label']
        description = data_frame['description']
        used_object_ids = data_frame['used_object_ids']
        object_tag2str = data_frame['object_tag2str']
        
        if label == "Action-True":
            label = 'yes'
        elif label in ["Action-False", "Distractor1", "Distractor2"]:
            label = 'no'
        else:
            raise NotImplementedError
        
        image = Image.open(image)
        # image = Image.open(BytesIO(base64.urlsafe_b64decode(image)))
        patch_image, bbox_target = self.patch_resize_transform(image, dict(boxes=torch.as_tensor(list(data_frame['rois_meta'].values()))[:,:4]))
        patch_mask = torch.tensor([True])

        if self.dataset.add_bbox_info:
            if description is None:
                description = ''
            for object_id, box_info in enumerate(bbox_target["boxes"]):
                if object_id in used_object_ids:
                    quant_x0 = "<bin_{}>".format(int((box_info[0] * (self.num_bins - 1)).round()))
                    quant_y0 = "<bin_{}>".format(int((box_info[1] * (self.num_bins - 1)).round()))
                    quant_x1 = "<bin_{}>".format(int((box_info[2] * (self.num_bins - 1)).round()))
                    quant_y1 = "<bin_{}>".format(int((box_info[3] * (self.num_bins - 1)).round()))
                    region_coord = "{} {} {} {}".format(quant_x0, quant_y0, quant_x1, quant_y1)
                    description += f" {object_tag2str[f'[{object_id}]']} is at {region_coord}."

        hypothesis = self.pre_caption(hypothesis, self.max_src_length)
        if description is None:
            src_item = self.encode_text(' does the image describe " {} "?'.format(hypothesis))
        else:
            src_item = self.encode_text(' we know " {} ". does the image describe " {} "?'.format(description, hypothesis))
            
        tgt_item = self.encode_text(" {}".format(label))
        ref_dict = {label: 1.0}

        if self.add_caption:  # TODO(HUI): try exchanging the text1 and text2 position
            caption = self.pre_caption(caption, self.max_src_length)
            if description is None:
                src_item = self.encode_text(' can image and text1 " {} " imply text2 " {} "?'.format(caption, hypothesis))
            else:
                src_item = self.encode_text(' we know " {} ". can image and text1 " {} " imply text2 " {} "?'.format(description, caption, hypothesis))

        src_item = torch.cat([self.bos_item, src_item, self.eos_item])
        if self.prompt_type == 'none':
            prev_output_item = torch.cat([self.bos_item, tgt_item])
            target_item = torch.cat([prev_output_item[1:], self.eos_item])
            decoder_prompt = self.bos_item
        elif self.prompt_type == 'src':
            prev_output_item = torch.cat([src_item, tgt_item])
            target_item = torch.cat([prev_output_item[1:], self.eos_item])
            decoder_prompt = src_item
        elif self.prompt_type == 'prev_output':
            prev_output_item = torch.cat([src_item[:-1], tgt_item])
            target_item = torch.cat([prev_output_item[1:], self.eos_item])
            decoder_prompt = src_item[:-1]
        else:
            raise NotImplementedError
        target_item[:-len(tgt_item)-1] = self.tgt_dict.pad()

        example = {
            "id": uniq_id,
            "img_id": img_id,
            "source": src_item,
            "patch_image": patch_image,
            "patch_mask": patch_mask,
            "target": target_item,
            "prev_output_tokens": prev_output_item,
            "decoder_prompt": decoder_prompt,
            "ref_dict": ref_dict,
            "ans_pos_idx": data_frame['ans_pos_idx'],
            "answer_label": data_frame['answer_label'],
        }
        if self.constraint_trie is not None:
            constraint_mask = torch.zeros((len(target_item), len(self.tgt_dict))).bool()
            start_idx = len(target_item) - len(tgt_item) - 1
            for i in range(len(target_item)-len(tgt_item)-1, len(target_item)):
                constraint_prefix_token = [self.tgt_dict.bos()] + target_item[start_idx:i].tolist()
                constraint_nodes = self.constraint_trie.get_next_layer(constraint_prefix_token)
                constraint_mask[i][constraint_nodes] = True
            example["constraint_mask"] = constraint_mask
        return example

    def collater(self, samples, pad_to_length=None):
        """Merge a list of samples to form a mini-batch.
        Args:
            samples (List[dict]): samples to collate
        Returns:
            dict: a mini-batch containing the data of the task
        """
        return collate(samples, pad_idx=self.pad, eos_idx=self.eos)
