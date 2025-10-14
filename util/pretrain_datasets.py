from copy import deepcopy
import os
from typing import List, Tuple
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset
import random
import tokenizers
import json


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class MultimodalBertDataset(Dataset):
    def __init__(
        self,
        data_root,
        transform,
        SR,
        split,
        max_caption_length: int = 100,
        out_path = False
    ):
        self.max_caption_length = max_caption_length
        self.data_root = data_root
        self.transform = transform
        self.image_ids, self.images_list, self.report_list, self.prompt_list, self.anatomy_list = self.read_csv(split)
        self.out_path = out_path
        self.split = split
        
        self.tokenizer = tokenizers.Tokenizer.from_file("data/mimic_wordpiece.json")
        self.idxtoword = {v: k for k, v in self.tokenizer.get_vocab().items()}
        self.tokenizer.enable_truncation(max_length=self.max_caption_length)
        self.tokenizer.enable_padding(length=self.max_caption_length)

        self.SR = SR

    def __len__(self):
        return len(self.images_list)
    
    def _text_process(self, text):
        tem = text.split('.')
        tem = [i.strip() + '. ' for i in tem]
        random.shuffle(tem)

        choice = random.randint(1, len(tem))
        report = ''
        for i in range(choice):
            if tem[i] != '.' and tem[i] != '. ' and tem[i] != 'None. ' and tem[i] != 'none. ' and tem[i] != '':
                report += tem[i].lower()
        if report == '' and tem != []:
            report = tem[0]

        return report
    
    def _random_mask(self,tokens):
        masked_tokens = deepcopy(tokens)  # [1, 100]
        for i in range(1, masked_tokens.shape[1]-1):
            if masked_tokens[0][i] == 0:
                break
            
            if masked_tokens[0][i-1] == 3 and self.idxtoword[masked_tokens[0][i].item()][0:2] == '##':
                masked_tokens[0][i] = 3
                continue
            
            if masked_tokens[0][i-1] != 3 and self.idxtoword[masked_tokens[0][i].item()][0:2] == '##':
                continue

            prob = random.random()
            if prob < 0:   # 0.5
                masked_tokens[0][i] = 3

        return masked_tokens

    def __getitem__(self, index):
        image = pil_loader(os.path.join(self.data_root, 'images', self.images_list[index]))
        image = self.transform(image)
        report = self.report_list[index]
        sent = self._text_process(report)
        prompt = self.prompt_list[index]
        anatomies = self.anatomy_list[index]
        image_id = self.image_ids[index]
        
        sent = '[CLS] ' + sent
        encoded = self.tokenizer.encode(sent)
        ids = torch.tensor(encoded.ids).unsqueeze(0)
        attention_mask = torch.tensor(encoded.attention_mask).unsqueeze(0)
        type_ids = torch.tensor(encoded.type_ids).unsqueeze(0)
        masked_ids = self._random_mask(ids)

        if self.out_path:
            return image, ids, attention_mask, type_ids, masked_ids, report, prompt, anatomies, image_id, self.images_list[index]
        else:
            return image, ids, attention_mask, type_ids, masked_ids, report, prompt, anatomies, image_id


    def read_csv(self, split):
        csv_path = os.path.join(self.data_root, f'ANA-MIMIC-{split}.csv')
        df = pd.read_csv(csv_path, sep=',')
        df['anatomies'] = df['anatomies'].apply(json.loads)
        df.fillna('', inplace=True)
        return df['image_id'], df["image_path"], df["report"], df['prompt'], df['anatomies']


    def collate_fn(self, instances: List[Tuple]):
        image_list, ids_list, attention_mask_list, type_ids_list, masked_ids_list, report_gen_list, prompt_list, anatomy_list, image_ids = [], [], [], [], [], [], [], [], []
        # flattern
        for b in instances:
            image, ids, attention_mask, type_ids, masked_ids, report_gen, prompt, anatomies, image_id = b
            image_list.append(image)
            ids_list.append(ids)
            attention_mask_list.append(attention_mask)
            type_ids_list.append(type_ids)
            masked_ids_list.append(masked_ids)
            report_gen_list.append(report_gen)
            prompt_list.append(prompt)
            anatomy_list.append(anatomies)
            image_ids.append(image_id)

        # stack
        image_stack = torch.stack(image_list)
        ids_stack = torch.stack(ids_list).squeeze()
        attention_mask_stack = torch.stack(attention_mask_list).squeeze()
        type_ids_stack = torch.stack(type_ids_list).squeeze()
        masked_ids_stack = torch.stack(masked_ids_list).squeeze()

        # sort and add to dictionary
        return_dict = {
            "image": image_stack,
            "labels": ids_stack,
            "attention_mask": attention_mask_stack,
            "type_ids": type_ids_stack,
            "ids": masked_ids_stack,
            "text_input": report_gen_list,
            "prompt": prompt_list,
            "anatomies": anatomy_list,
            "image_ids": image_ids,
        }

        return return_dict
    
    
