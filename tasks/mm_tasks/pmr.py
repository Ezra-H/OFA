# Copyright 2022 The OFA-Sys Team. 
# All rights reserved.
# This source code is licensed under the Apache 2.0 license 
# found in the LICENSE file in the root directory.

import json
import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import os
import torch
from fairseq import metrics
from fairseq.tasks import register_task

from data.mm_data.pmr_base import PMRDatasetForOFA
from data.mm_data.pmr_dataset import PMRDataset
from tasks.ofa_task import OFAConfig, OFATask
from data.mm_data.snli_ve_dataset import SnliVeDataset
from data.file_dataset import FileDataset
from data import data_utils
from utils.trie import Trie

logger = logging.getLogger(__name__)


@dataclass
class PMRConfig(OFAConfig):
    ans2label_dict: Optional[str] = field(
        default='{"no": 0, "yes":1}',
        metadata={"help": 'answer to label dict'},
    )
    data_root: Optional[str] = field(
        default='/data5/huangrunhui/proj7/pmr_hcp/dataset/',
        metadata={"help": "add caption to encoder"},
    )
    img_dir: Optional[str] = field(
        default='images',
        metadata={"help": "image sub directory"},
    )
    use_adv: bool = field(
        default=False,
        metadata={"help": "use adversarial data"},
    )
    use_syn_caption: bool = field(
        default=False,
        metadata={"help": "use blip-generated caption to encoder"},
    )
    syn_caption_root: Optional[str] = field(
        default='/data5/huangrunhui/proj7/pmr_hcp/dataset/blip_caption',
        metadata={"help": "use blip-generated caption to encoder"},
    )
    valid_batch_size: int = field(
        default=20,
        metadata={"help": "valid batch size per step"},
    )
    prompt_type: Optional[str] = field(
        default='none',
        metadata={"help": "prompt_type"},
    )
    add_caption: bool = field(
        default=False,
        metadata={"help": "add caption to encoder"},
    )

@register_task("pmr", dataclass=PMRConfig)
class PMRTask(OFATask):
    def __init__(self, cfg: PMRConfig, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)
        self.ans2label_dict = json.loads(self.cfg.ans2label_dict)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        paths = self.cfg.data.split(',')
        assert len(paths) > 0
        print(split)
        split2file = dict(train='train-ori.jsonl',
                          val='val-ori.jsonl',
                          test='test-ori-without-label.jsonl')
        
        split2adv_file = dict(train='train-adv.jsonl',
                              val='val-adv.jsonl',
                              test='test-ori-without-label.jsonl')  # TODO(HUI): test not release.

        # if split == 'train':
        #     file_path = paths[(epoch - 1) % (len(paths) - 1)]
        # else:
        #     file_path = paths[-1]
        # dataset = FileDataset(file_path, self.cfg.selected_cols)

        split_pmr = 'val' if split == 'valid' else split
        dataset = PMRDatasetForOFA(
            img_db=os.path.join(self.cfg.data_root, self.cfg.img_dir),
            anno_dir=os.path.join(self.cfg.data_root, split2file[split_pmr]),
            split=split_pmr,
            use_adv=self.cfg.use_adv,
            use_syn_caption=self.cfg.use_syn_caption,
            syn_caption_root=self.cfg.syn_caption_root)

        self.datasets[split] = PMRDataset(
            split,
            dataset,
            self.bpe,
            self.src_dict,
            self.tgt_dict,
            max_src_length=self.cfg.max_src_length,
            max_tgt_length=self.cfg.max_tgt_length,
            patch_image_size=self.cfg.patch_image_size,
            add_caption=self.cfg.add_caption,
            constraint_trie=self.constraint_trie,
            imagenet_default_mean_and_std=self.cfg.imagenet_default_mean_and_std,
            prompt_type=self.cfg.prompt_type
        )

    def build_model(self, cfg):
        model = super().build_model(cfg)
        answer_item_list = []
        self.index2ans = {}
        self.constraint_trie = Trie(self.tgt_dict.eos())
        for i, answer in enumerate(self.ans2label_dict.keys()):
            answer_item = self.tgt_dict.encode_line(
                line=self.bpe.encode(' ' + answer),
                add_if_not_exist=False,
                append_eos=False
            ).long()
            answer_item_list.append(answer_item)
            self.index2ans[i] = answer
            self.constraint_trie.insert([self.tgt_dict.bos()] + answer_item.tolist() + [self.tgt_dict.eos()])

        constraint_mask_list = []
        for answer_item in answer_item_list:
            constraint_mask = torch.zeros((len(answer_item)+1, len(self.tgt_dict))).bool()
            for i in range(len(answer_item)+1):
                constraint_prefix_token = [self.src_dict.bos()] + answer_item[:i].tolist()
                constraint_nodes = self.constraint_trie.get_next_layer(constraint_prefix_token)
                constraint_mask[i][constraint_nodes] = True
            constraint_mask_list.append(constraint_mask)

        self.valid_answers_list = []
        self.valid_constraint_masks_list = []
        for i in range(0, len(answer_item_list), self.cfg.valid_batch_size):
            self.valid_answers_list += [answer_item_list[i:i+self.cfg.valid_batch_size]]
            self.valid_constraint_masks_list += [constraint_mask_list[i:i+self.cfg.valid_batch_size]]

        return model

    def build_generator(
        self, models, args, seq_gen_cls=None, extra_gen_cls_kwargs=None, prefix_allowed_tokens_fn=None,
    ):
        seq_generator = super().build_generator(models, args, seq_gen_cls, extra_gen_cls_kwargs, prefix_allowed_tokens_fn)
        seq_generator.constraint_trie = self.constraint_trie

        return seq_generator

    def valid_step(self, sample, model, criterion, **extra_kwargs):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)

        model.eval()
        with torch.no_grad():
            encoder_out = model.encoder(
                sample["net_input"]["src_tokens"],
                src_lengths=sample["net_input"]["src_lengths"],
                patch_images=sample["net_input"]["patch_images"],
                patch_masks=sample["net_input"]["patch_masks"]
            )
            device = sample["net_input"]["src_tokens"].device
            eos_item = torch.tensor([self.src_dict.eos()])
            pad = self.src_dict.pad()
            valid_result = []
            for valid_answers, valid_constraint_masks in zip(self.valid_answers_list, self.valid_constraint_masks_list):
                valid_size = len(valid_answers)
                valid_tgt_items = [
                    torch.cat([torch.tensor(decoder_prompt[1:]), valid_answer, eos_item])
                    for decoder_prompt in sample["decoder_prompts"] for valid_answer in valid_answers
                ]
                valid_prev_items = [
                    torch.cat([torch.tensor(decoder_prompt), valid_answer])
                    for decoder_prompt in sample["decoder_prompts"] for valid_answer in valid_answers
                ]
                valid_constraint_mask_items = [
                    torch.cat([torch.zeros(len(decoder_prompt)-1, valid_constraint_mask.size(1)).bool(), valid_constraint_mask], dim=0)
                    for decoder_prompt in sample["decoder_prompts"] for valid_constraint_mask in valid_constraint_masks
                ]
                valid_tgt = data_utils.collate_tokens(valid_tgt_items, pad_idx=pad, left_pad=False).to(device)
                valid_prev_output = data_utils.collate_tokens(valid_prev_items, pad_idx=pad, left_pad=False).to(device)
                valid_constraint_masks = data_utils.collate_tokens(valid_constraint_mask_items, pad_idx=pad, left_pad=False).to(device)

                new_encoder_out = {}
                new_encoder_out["encoder_out"] = [
                    encoder_out["encoder_out"][0].repeat_interleave(valid_size, dim=1)
                ]
                new_encoder_out["encoder_padding_mask"] = [
                    encoder_out["encoder_padding_mask"][0].repeat_interleave(valid_size, dim=0)
                ]
                new_encoder_out["position_embeddings"] = [
                    encoder_out["position_embeddings"][0].repeat_interleave(valid_size, dim=0)
                ]

                decoder_out = model.decoder(valid_prev_output, encoder_out=new_encoder_out)
                decoder_out[0].masked_fill_(~valid_constraint_masks, -math.inf)
                lprobs = model.get_normalized_probs(decoder_out, log_probs=True)
                scores = lprobs.gather(dim=-1, index=valid_tgt.unsqueeze(-1)).squeeze(-1)
                scores = scores.masked_fill(valid_tgt.eq(self.tgt_dict.pad()), 0)
                scores = scores.masked_fill((~valid_constraint_masks).all(2), 0)
                scores = scores.sum(1)
                scores = scores.view(-1, valid_size)
                valid_result.append(scores)

        valid_result = torch.cat(valid_result, dim=-1)
        predicts = valid_result.argmax(1).tolist()
        hyps = [self.index2ans[predict_index] for predict_index in predicts]
        scores = [ref_dict.get(hyp, 0) for ref_dict, hyp in zip(sample['ref_dict'], hyps)]
        logging_output["_pmr_score_sum"] = sum(scores)
        logging_output["_pmr_cnt"] = len(scores)

        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)

        def sum_logs(key):
            import torch
            result = sum(log.get(key, 0) for log in logging_outputs)
            if torch.is_tensor(result):
                result = result.cpu()
            return result

        def compute_score(meters):
            score = meters["_pmr_score_sum"].sum / meters["_pmr_cnt"].sum
            score = score if isinstance(score, float) else score.item()
            return round(score, 4)

        if sum_logs("_snli_cnt") > 0:
            metrics.log_scalar("_pmr_score_sum", sum_logs("_pmr_score_sum"))
            metrics.log_scalar("_pmr_cnt", sum_logs("_pmr_cnt"))
            metrics.log_derived("pmr_score", compute_score)
