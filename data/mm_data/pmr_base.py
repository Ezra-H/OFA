import os
import random
import pickle
from collections import defaultdict

import json
import jsonlines
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from transformers import AutoTokenizer
from PIL import Image
import multiprocessing

from tasks.mm_tasks.dist import broadcast, get_world_size, get_rank

max_workers = multiprocessing.cpu_count()

torch.multiprocessing.set_sharing_strategy('file_system')

GENDER_NEUTRAL_NAMES = ['Casey', 'Riley', 'Jessie', 'Jackie', 'Avery', 'Jaime', 'Peyton', 'Kerry', 'Jody', 'Kendall',
                        'Frankie', 'Pat', 'Quinn']


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


def read_json(fname):
    with open(fname, 'r') as f:
        return json.load(f)


class PMRDatasetReturnImg(Dataset):
    def __init__(self, img_db=None, anno_dir=None,
                 transform=None,
                 split='train', use_adv=False,
                 use_syn_caption=False, syn_caption_root=None,
                 whole_img_first=False,
                 add_bbox_info=False, num_bins=1000):
        super(PMRDatasetReturnImg, self).__init__()
        
        self.img_db = img_db
        
        assert split in ['train', 'val', 'test']
        self.split = split
        
        self.transform = transform
        
        self.annos = read_jsonl(anno_dir)
        
        adv_map = dict(train='train-adv.jsonl', val='val-adv.jsonl')
        
        self.whole_img_first=whole_img_first
        
        self.add_bbox_info = add_bbox_info
        self.num_bins=num_bins
        
        self.use_adv = use_adv
        self.use_syn_caption = use_syn_caption
        self.syn_caption_root = syn_caption_root
        self.syn_captions = None
        self.syn_captions_adv = None
        if self.use_syn_caption:
            self.syn_captions = read_json(
                os.path.join(self.syn_caption_root, os.path.split(anno_dir)[1].replace('jsonl', 'json')))
        
        self.annos_adv = None
        if self.use_adv and split in ['train', 'val']:
            self.annos_adv = read_jsonl(os.path.join(os.path.split(anno_dir)[0], adv_map[split]))
            
            if self.use_syn_caption:
                self.syn_captions_adv = read_json(
                    os.path.join(self.syn_caption_root, adv_map[split].replace('jsonl', 'json')))
        
        self.person_name_id = 0
        
        self.classes = ["Action-True", "Action-False", "Distractor1", "Distractor2"]
        self.category_list = ['personality', 'identity', 'mood', 'antecedent', 'environment', 'character', 'surroundings', 'relationship']
    
        self.prepare_data()

    def process_syn_caption(self, used_object_ids, syn_captions, object_map):
        # used_object_list = []
        # for mix_token in premise:
        #     if isinstance(mix_token, list):
        #         used_object_list.append(mix_token[0])
        # for one_answer in answer_choices:
        #     for mix_token in one_answer:
        #         if isinstance(mix_token, list):
        #             used_object_list.append(mix_token[0])
        #
        # used_object_list = np.asarray(list(set(used_object_list)))

        split_syn_captions = []
        for syn_caption in syn_captions:
            syn_caption = syn_caption.replace('a close up of ', '')
            syn_caption = syn_caption.replace('a blurry photo of ', '')
            syn_caption = syn_caption.replace('a blurry picture of ', '')
            split_syn_captions.append(syn_caption)
    
        description = ''
        for idx, caption in enumerate(split_syn_captions):
            if idx in used_object_ids:
                object_name = object_map[f"[{idx}]"]
                description += f' {object_name} is {caption}.'
    
        return description.strip()

    def prepare_data(self):
        self.examples = []
        for i, data in enumerate(self.annos):
            img_fn = data['img_fn'].replace('.../', '.__/')  # invalid path
            img_fn = img_fn.replace('./', '/')
            img_dir = os.path.join(self.img_db, img_fn)

            metadata_fn = data['metadata_fn'].replace('.../', '.__/')  # invalid path
            metadata_fn = metadata_fn.replace('./', '/')
            metadata_dir = os.path.join(self.img_db, metadata_fn)
            metadata = read_json(metadata_dir)
            rois_meta = dict(zip(metadata['names'], metadata['boxes']))
            
            total_id = data["total_id"]
            img_id = data["img_id"]
        
            raw_object_names = objects = metadata['names']
            objects = list(map(self.convert_person_to_gender_neutral_name, objects))
            object_map = {f'[{i}]': o for i, o in enumerate(objects)}
        
            premise = self._convert_tokens(data['premise'], object_map)

            # prepare the used object ids.
            used_object_ids = []
            for mix_token in premise:
                if isinstance(mix_token, list):
                    used_object_ids.append(mix_token[0])
            for one_answer in data['answer_choices']:
                for mix_token in one_answer:
                    if isinstance(mix_token, list):
                        used_object_ids.append(mix_token[0])
            used_object_ids = np.asarray(list(set(used_object_ids)))

            description = None
            if self.use_syn_caption:  # TODO(HUI): use shuffle to diverse the input.
                description = self.process_syn_caption(
                    # data['premise'], data['answer_choices'],
                    used_object_ids,
                    self.syn_captions[str(total_id)], object_map)

                # premise = ' '.join([f'{o} is {t}.' for ro, o, t in
                #                     zip(raw_object_names, objects, self.syn_captions[str(total_id)]) if ro == 'person'])
        
            for j, answer in enumerate(data['answer_choices']):
                answer = self._convert_tokens(answer, object_map)
            
                # 4-class classification.
                target = -1
                if 'answer_types' in data.keys():
                    target = self.classes.index(data['answer_types'][j])
                
                category_id = self.category_list.index(data['category'])

                # ITM task.
                is_answer = -1
                answer_label = -1
                label='Action-False'
                if 'answer_label' in data:
                    is_answer = j == data['answer_label']
                    answer_label = data['answer_label']  # for eval
                    label = data['answer_types'][j]

                self.examples.append(
                    dict(img_dir=img_dir, premise=premise, answer=answer,
                         label=label,
                         rois_meta=rois_meta, objects=objects,
                         target=target, is_answer=is_answer, total_id=total_id,
                         img_id=img_id, ans_pos_idx=j, answer_label=answer_label,
                         category_id=category_id, description=description,
                         used_object_ids=used_object_ids, object_tag2str=object_map,
                         )
                )
            
                # self.examples.append(
                #     [img_dir, premise, answer,
                #      rois_meta, objects,
                #      target, is_answer, total_id, img_id, j, answer_label])
            
                if self.annos_adv is not None:
                    # adversarial sample. only 'premise' and 'answer' texts are different.
                    adv_data = self.annos_adv[i]
                    adv_premise = self._convert_tokens(adv_data['premise'], object_map)  # same objects
                    adv_answer = self._convert_tokens(adv_data['answer_choices'][j], object_map)
                
                    # if self.use_syn_caption:  # TODO(HUI): use shuffle to diverse the input.
                        # adv_premise = ' '.join(
                        #     [adv_premise] + [f'{o} is {t}.' for ro, o, t in
                        #                      zip(raw_object_names, objects, self.syn_captions_adv[str(total_id)]) if
                        #                      ro == 'person'])
                        # description = self.process_syn_caption(data['premise'], data['answer_choices'],
                        #                                        self.syn_captions[str(total_id)])

                        # adv_premise = self.process_syn_caption(premise,
                        #                                    data['answer_choices'],
                        #                                    self.syn_captions[str(data['total_id'])])
                        
                    self.examples.append(
                        dict(img_dir=img_dir, premise=adv_premise, answer=adv_answer,
                             rois_meta=rois_meta, objects=objects,
                             label=data['answer_types'][j],
                             target=target, is_answer=is_answer, total_id=total_id,
                             img_id=img_id, ans_pos_idx=j, answer_label=answer_label,
                             category_id=category_id,
                             description=description)
                    )
                    
                    # self.examples.append(
                    #     [img_dir, adv_premise, adv_answer,
                    #      rois_meta, objects,
                    #      target, is_answer, total_id, img_id, j, answer_label])
    
    def convert_person_to_gender_neutral_name(self, object_name):
        if object_name == 'person':
            objects_replace_name = GENDER_NEUTRAL_NAMES[self.person_name_id]
            self.person_name_id = (self.person_name_id + 1) % len(GENDER_NEUTRAL_NAMES)
            return objects_replace_name
            # return random.choice(self.gender_neutral_names)
        else:
            return object_name
    
    def __len__(self):
        return len(self.examples)

    def crop_roi(self,
                 image,
                 rois,
                 expand=1.5,
                 ):
        assert expand > 1
        bboxs_img = []
        positions = []
    
        w_image, h_image = image.size

        if self.whole_img_first:
            image_tf = self.transform(image)
            bboxs_img.append(image_tf)
            positions.append(torch.tensor([0., 0., w_image, h_image, w_image, h_image, w_image * h_image]))

        for (x1, y1, x2, y2, s) in rois.values():
        
            w_org, h_org = (x2 - x1), (y2 - y1)
        
            if w_org * h_org < 3000:
                w, h = (x2 - x1) * expand, (y2 - y1) * expand
                dw = w - w_org
                dh = h - h_org
            
                x1, y1, x2, y2 = max(0, x1 - dw / 2), max(0, y1 - dh / 2), min(x2 + dw / 2, w_image), min(y2 + dh / 2,
                                                                                                          h_image)
            else:
                w, h = (x2 - x1), (y2 - y1)
        
            # x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # maybe dont need int data.
        
            box = image.crop((x1, y1, x2, y2))
            if self.transform:
                box = self.transform(box)

            bboxs_img.append(box)
            positions.append(torch.tensor([x1, y1, x2, y2, w, h, w * h]))

        if not self.whole_img_first:
            image_tf = self.transform(image)
            bboxs_img.append(image_tf)
            positions.append(torch.tensor([0., 0., w_image, h_image, w_image, h_image, w_image * h_image]))
    
        bboxs_img = torch.stack(bboxs_img, dim=0)
        positions = torch.stack(positions, dim=0)
    
        return bboxs_img, positions
    
    def __getitem__(self, index):
        # (img_dir, premise, answer,
        #  rois_meta, objects,
        #  target, is_answer, total_id, img_id, ans_pos_idx, answer_label) = self.examples[index]

        data = self.examples[index]
        img_dir = data.pop('img_dir')
        premise = data.pop('premise')
        answer = data.pop('answer')
        rois_meta = data.pop('rois_meta')

        img = Image.open(img_dir)
        roi, position = self.crop_roi(img, rois_meta)
    
        # (img_dir, premise, answer,
        #  roi, position, objects,
        #  target, is_answer, total_id, img_id, ans_pos_idx, answer_label) = self.get_one_example(index)
        
        raise NotImplementedError
        tokenized = self.tokenizer(f"Premise: {premise} Answer: {answer}",
                                   padding="max_length",
                                   truncation=True,
                                   return_tensors="pt")
        input_ids = tokenized['input_ids'][0]
        token_type_ids = tokenized['token_type_ids'][0]
        text_attention_mask = tokenized['attention_mask'][0]
        attention_mask = torch.cat((text_attention_mask, torch.ones(len(roi), dtype=torch.long)), dim=0)

        data['input_ids'] = input_ids
        data['token_type_ids'] = token_type_ids
        data['attention_mask'] = attention_mask
        data['roi'] = roi
        data['position'] = position

        return data
        # dict
        # return [input_ids, token_type_ids, attention_masks,
        #         roi, position, target, is_answer,
        #         total_id, img_id, ans_pos_idx, answer_label]

    def collate(self, inputs):
        # (input_ids, token_type_ids, attention_masks,
        #  rois, positions, targets, is_answers,
        #  total_ids, img_ids, ans_pos_idxs, answer_labels) = map(list, unzip(inputs))
        
        collections = defaultdict(list)
        for d in inputs:
            for k, v in d.items():
                collections[k].append(v)
        
        input_ids = torch.stack(collections['input_ids'], dim=0)
        token_type_ids = torch.stack(collections['token_type_ids'], dim=0)
        text_lens = [i.size(0) for i in input_ids]
        batch_size, max_text_length = input_ids.size()
        # input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        text_position_ids = torch.arange(0, max(text_lens), dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
        attention_masks = pad_sequence(collections['attention_mask'], batch_first=True, padding_value=0)
        targets = torch.tensor(collections['target'], dtype=torch.long)
        is_answers = torch.tensor(collections['is_answer'], dtype=torch.long)
        category_ids = torch.tensor(collections['category_id'], dtype=torch.long)

        ans_pos_idxs = torch.tensor(collections['ans_pos_idx'], dtype=torch.long)
        answer_labels = torch.tensor(collections['answer_label'], dtype=torch.long)
        
        num_bboxes = [roi.size(0) for roi in collections['roi']]
        roi = self._pad_tensors(collections['roi'], num_bboxes)
        position = self._pad_tensors(collections['position'], num_bboxes)
    
        out_size = attention_masks.size(1)
        gather_index = self.get_gather_index(text_lens, num_bboxes, batch_size, max_text_length, out_size)
    
        batch = dict(
            total_ids=collections['total_id'],
            img_ids=collections['img_id'],
            input_ids=input_ids,
            text_position_ids=text_position_ids,
            token_type_ids=token_type_ids,
            roi=roi,
            position=position,
            attention_masks=attention_masks,
            gather_index=gather_index,
            targets=targets,
            is_answers=is_answers,
            ans_pos_idxs=ans_pos_idxs,
            answer_labels=answer_labels,
            category_ids=category_ids
        )
        return batch

    def _convert_tokens(self, token_list, object_map):
        if isinstance(token_list, str):
            return token_list
        elif isinstance(token_list, list):
            token_list = list(map(str, token_list))
            new_token_list = []
            for token in token_list:
                if token in object_map.keys():
                    new_token_list.append(object_map[token])
                else:
                    new_token_list.append(token)
            return ' '.join(new_token_list)
    
    def _pad_tensors(self, tensors, lens=None, pad=0):
        """B x [T, ...]"""  # B, T, 7
        if lens is None:
            lens = [t.size(0) for t in tensors]
        max_len = max(lens)
        bs = len(tensors)
        hid = tensors[0][0].size()
        dtype = tensors[0].dtype
        output = torch.zeros((bs, max_len) + hid, dtype=dtype)
        if pad:
            output.data.fill_(pad)
        for i, (t, l) in enumerate(zip(tensors, lens)):
            output.data[i, :l, ...] = t.data
        return output
    
    def get_gather_index(self, text_lens, num_bboxes, batch_size, max_text_length, out_size):
        assert len(text_lens) == len(num_bboxes) == batch_size
        gather_index = torch.arange(0, out_size, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
        
        for i, (tl, nbb) in enumerate(zip(text_lens, num_bboxes)):
            gather_index.data[i, tl:tl + nbb] = torch.arange(max_text_length, max_text_length + nbb,
                                                             dtype=torch.long).data
        return gather_index


class PMRDatasetForOFA(PMRDatasetReturnImg):
    def __init__(self, *args, **kwargs):
        super(PMRDatasetForOFA, self).__init__(*args, **kwargs)
        
        self.total_row_count = len(self.examples)
        self.slice_id = get_rank()
        self.slice_count = get_world_size()
        self.row_count = self.total_row_count // self.slice_count
        
        if self.slice_id == 0:
            slice_list = [r * self.row_count for r in range(self.slice_count)]
            random.shuffle(slice_list)
            slice_list = broadcast(slice_list)
        else:
            slice_list = broadcast(None)

        self.slice_start_idx = slice_list[self.slice_id]
        self.examples = self.examples[self.slice_start_idx:self.slice_start_idx+self.row_count]
        print("slice_id {} slice num {} total row count {}".format(
            self.slice_id, self.row_count, self.total_row_count)
        )
    
    def _seek(self, offset=0):
        pass
    
    def get_total_row_count(self):
        return self.total_row_count

    def __getitem__(self, index):
        data = self.examples[index]
        # img_dir = data.pop('img_dir')
        # question = data.pop('premise')
        # answer = data.pop('answer')
        # rois_meta = data.pop('rois_meta')
        
        # img = Image.open(img_dir)
        # roi, position = self.crop_roi(img, rois_meta)
        #
        # object_mask = torch.ones(len(roi), dtype=torch.long)
        
        # question = f'Premise: {question}'
        # answer = f'Answer: {answer}'
        
        # data['object_mask'] = object_mask
        # data['question'] = question
        # data['answer'] = answer
        
        return data
        # return [question, answer,
        #         roi, position, object_mask, target, is_answer,
        #         total_id, img_id, ans_pos_idx, answer_label]
    
    # def collate(self, inputs):
    #     # (questions, answers,
    #     #  rois, positions, object_mask, targets, is_answers,
    #     #  total_ids, img_ids, ans_pos_idxs, answer_labels) = map(list, unzip(inputs))
    #
    #     collections = defaultdict(list)
    #     for d in inputs:
    #         for k, v in d.items():
    #             collections[k].append(v)
    #
    #     # input_ids = torch.stack(input_ids, dim=0)
    #     # token_type_ids = torch.stack(token_type_ids, dim=0)
    #     # text_lens = [i.size(0) for i in input_ids]
    #     # batch_size, max_text_length = input_ids.size()
    #     # input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    #     # text_position_ids = torch.arange(0, max(text_lens), dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
    #     object_mask = pad_sequence(collections['object_mask'], batch_first=True, padding_value=0)
    #
    #     targets = torch.tensor(collections['target'], dtype=torch.long)
    #     is_answers = torch.tensor(collections['is_answer'], dtype=torch.long)
    #
    #     ans_pos_idxs = torch.tensor(collections['ans_pos_idx'], dtype=torch.long)
    #     answer_labels = torch.tensor(collections['answer_label'], dtype=torch.long)
    #
    #     num_bboxes = [roi.size(0) for roi in collections['roi']]
    #     rois = self._pad_tensors(collections['roi'], num_bboxes)
    #     positions = self._pad_tensors(collections['position'], num_bboxes)
    #
    #     # out_size = attention_masks.size(1)
    #     # gather_index = self.get_gather_index(text_lens, num_bboxes, batch_size, max_text_length, out_size)
    #
    #     batch = dict(
    #         questions=collections['question'],
    #         answers=collections['answer'],
    #         rois=rois,
    #         positions=positions,
    #         # gather_index=gather_index,
    #         object_mask=object_mask,
    #         targets=targets,
    #         is_answers=is_answers,
    #         total_ids=collections['total_id'],
    #         img_ids=collections['img_id'],
    #         ans_pos_idxs=ans_pos_idxs,
    #         answer_labels=answer_labels,
    #     )
    #     return batch


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    train = PMRDataset(img_db="dataset/images/", anno_dir="dataset/train-ori.jsonl", tokenizer=tokenizer)
    print(train[0])
    print(len(train))
