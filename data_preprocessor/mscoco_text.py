
from argparse import ArgumentParser
import json
from pathlib import Path
import pdb
# import pandas as pd
import os
import random
from os.path import exists
from tqdm import tqdm

class InstructData:

    def __init__(self, args):
        self.annotation_fn = './raw_datasets/coco_text/COCO_Text.json'
        self.train_img_dir = Path('./raw_datasets/MSCOCO2014/train2014')
        self.val_img_dir = Path('./raw_datasets/MSCOCO2014/val2014')
        self.task_type = args.task_type
        self.out_data_dir = args.out_data_dir
        self.out_data_dir.mkdir(parents=True, exist_ok=True)
        self.meta_info_fn = 'meta.json'
        self.num_train = args.num_train
        self.num_val = args.num_val
        self.vocabs = [['Yes','No'],['yes','no'],['True',"False"],['yes, the text matches the text in the region','no, the text is different from the text in the region']]

    def get_img_fn(self, file_name):
        if 'train' in file_name:
            return self.train_img_dir / file_name
        elif 'val' in file_name:
            return self.val_img_dir / file_name
        else:
            raise NotImplemented
    
    def create_text_localization_data(self):
        """
        Given the image and the text, select the region that contains the exact text information.
        """
            
        data_dict = {
            'train': [],
            'val': []
        }
        with open(self.annotation_fn, 'r') as f:
            data = json.load(f)
            
        for img_id, anns in tqdm(data['imgToAnns'].items(), desc='processing data'):
            if anns:
                img_fn = self.get_img_fn(data['imgs'][img_id]['file_name'])
                exists(img_fn)
                split = data['imgs'][img_id]['set']
                img_id = data['imgs'][img_id]['id']
                regions = []
                texts = []
                for ann in anns:
                    ann_data = data['anns'][str(ann)]
                    bbox = [ann_data['bbox'][0],ann_data['bbox'][1], ann_data['bbox'][0]+ann_data['bbox'][2], ann_data['bbox'][1]+ann_data['bbox'][3]]
                    if 'utf8_string' in ann_data:
                        if (bbox[3] - bbox[1]) * (bbox[2] - bbox[0]) >= 500 * len(ann_data['utf8_string']):
                            regions.append(bbox)
                            texts.append(ann_data['utf8_string'])
                
                if len(texts) > 1 and len(regions) > 1:
                    for text, region in zip(texts, regions):
                        out_dict = {'unique_id': f'mscoco_{self.task_type}_{img_id}',
                                    'image_source': 'coco2014',
                                    'task_name':  self.task_type,
                                    'options': regions,
                                    'image_path': str(img_fn),
                                    'region': [region],
                                    'text': text
                        }
                        data_dict[split].append(out_dict)
        
        return data_dict    

    def create_region_text_match_data(self):
        """
        Given a text region and extra text information, determine whether the text information matches text appeared in the region.
        """
            
        data_dict = {
            'train': [],
            'val': []
        }
        with open(self.annotation_fn, 'r') as f:
            data = json.load(f)
            
        for img_id, anns in tqdm(data['imgToAnns'].items(), desc='processing data'):
            if anns:
                img_fn = self.get_img_fn(data['imgs'][img_id]['file_name'])
                exists(img_fn)
                split = data['imgs'][img_id]['set']
                img_id = data['imgs'][img_id]['id']
                regions = []
                texts = []
                for ann in anns:
                    ann_data = data['anns'][str(ann)]
                    bbox = [ann_data['bbox'][0],ann_data['bbox'][1], ann_data['bbox'][0]+ann_data['bbox'][2], ann_data['bbox'][1]+ann_data['bbox'][3]]
                    if 'utf8_string' in ann_data:
                        if (bbox[3] - bbox[1]) * (bbox[2] - bbox[0]) >= 500 * len(ann_data['utf8_string']):
                            regions.append(bbox)
                            texts.append(ann_data['utf8_string'])
                
                if len(texts) > 1 and len(regions) > 1:
                    for idx in range(len(texts)):
                        vocab = random.choice(self.vocabs)
                        if random.uniform(0, 1) > 0.5: # match
                            out_dict = {'unique_id': f'mscoco_{self.task_type}_{img_id}',
                                        'image_source': 'coco2014',
                                        'task_name':  self.task_type,
                                        'image_path': str(img_fn),
                                        'region': [regions[idx]],
                                        'text': texts[idx],
                                        'options': vocab,
                                        'target_txt': vocab[0]
                            }
                        else:
                            idx_lst = list(range(len(texts)))
                            other_idx = random.choice(idx_lst[:idx] + idx_lst[idx+1:])
                            out_dict = {'unique_id': f'mscoco_{self.task_type}_{img_id}',
                                        'image_source': 'coco2014',
                                        'task_name':  self.task_type,
                                        'image_path': str(img_fn),
                                        'region': [regions[idx]],
                                        'text': texts[other_idx],
                                        'options': vocab,
                                        'target_txt': vocab[1]
                            }
                        
                        data_dict[split].append(out_dict)
        
        return data_dict    

    def create_text_legibility_data(self):
        """
        Given a text region in the image, detect whether the text is legible.
        """
        
        data_dict = {
            'train': [],
            'val': []
        }
        with open(self.annotation_fn, 'r') as f:
            data = json.load(f)
        
        given_options = [x['name'] for x in list(data['cats']['legibility'].values())]
        vocabs = [['clear','unclear'], ['clear','not clear and complete'],['yes, it is clear','not, it is not clear'],['legible','illegible']]
        
        options = random.choice(vocabs)    
        for img_id, anns in tqdm(data['imgToAnns'].items(), desc='processing data'):
            if anns:
                img_fn = self.get_img_fn(data['imgs'][img_id]['file_name'])
                exists(img_fn)
                split = data['imgs'][img_id]['set']
                img_id = data['imgs'][img_id]['id']
                for ann in anns:
                    ann_data = data['anns'][str(ann)]
                    bbox = [ann_data['bbox'][0],ann_data['bbox'][1], ann_data['bbox'][0]+ann_data['bbox'][2], ann_data['bbox'][1]+ann_data['bbox'][3]]
                    out_dict = {'unique_id': f'mscoco_{self.task_type}_{img_id}',
                                'image_source': 'coco2014',
                                'task_name':  self.task_type,
                                'image_path': str(img_fn),
                                'region': [bbox],
                                'options': options,
                                'target_txt': options[0] if ann_data['legibility'] == 'legible' else options[1]
                    }
                    data_dict[split].append(out_dict)

        return data_dict  

    def create_text_type_data(self):
        """
        Given a text region in the image, tell me whether the text is machine-printed, handwritten, or other.
        """
        
        data_dict = {
            'train': [],
            'val': []
        }
        with open(self.annotation_fn, 'r') as f:
            data = json.load(f)
        
        options = [x['name'] for x in list(data['cats']['class'].values())]
            
        for img_id, anns in tqdm(data['imgToAnns'].items(), desc='processing data'):
            if anns:
                img_fn = self.get_img_fn(data['imgs'][img_id]['file_name'])
                exists(img_fn)
                split = data['imgs'][img_id]['set']
                img_id = data['imgs'][img_id]['id']
                for ann in anns:
                    ann_data = data['anns'][str(ann)]
                    bbox = [ann_data['bbox'][0],ann_data['bbox'][1], ann_data['bbox'][0]+ann_data['bbox'][2], ann_data['bbox'][1]+ann_data['bbox'][3]]
                    out_dict = {'unique_id': f'mscoco_{self.task_type}_{img_id}',
                                'image_source': 'coco2014',
                                'task_name':  self.task_type,
                                'image_path': str(img_fn),
                                'region': [bbox],
                                'options': options,
                                'target_txt': ann_data['class']
                    }
                    data_dict[split].append(out_dict)
        
        return data_dict  

    def create_data(self):

        if self.task_type == 'text_localization':
            # Given the image and the text, select the region that contains the exact text information.
            data_dict = self.create_text_localization_data()
        elif self.task_type == 'region_text_match':
            # Given a text region and extra text information, determine whether the text information matches text appeared in the region.
            data_dict = self.create_region_text_match_data()
        elif self.task_type == 'text_legibility':
            data_dict = self.create_text_legibility_data()
            # Given a text region in the image, detect whether the text is legible.
        elif self.task_type == 'text_type':
            # Given a text region in the image, tell me whether the text is machine-printed, handwritten, or other.
            data_dict = self.create_text_type_data()
        else:
            raise NotImplemented
            
        for split, data in data_dict.items():
            if split == 'train':
                data = data[:self.num_train]
            elif split == 'val':
                data = data[:self.num_val]
            print(f'{self.task_type}[{split}] = {len(data)}')
            save_path = self.out_data_dir / f'{split}.jsonl' if split == 'train' else self.out_data_dir / 'valid.jsonl'
            with open(save_path,'w') as fout:
                for ex in data:
                    fout.write(json.dumps(ex)+'\n')
           

if __name__ == '__main__':
    arg_parser = ArgumentParser(
        'Create dataset for COCO Text')
    arg_parser.add_argument('--out_data_dir', type=Path,
                            default='instruct_data',
                            help='Path to the output data folder')
    arg_parser.add_argument('--task_type', type=str,
                            default='region_text_match',
                            choices=['text_localization', 'region_text_match', 'text_legibility', 'text_type'],
                            help='Specify the type of task to create dataset')
    arg_parser.add_argument('--num_train', type=int,
                            help='number of training instance')
    arg_parser.add_argument('--num_val', type=int,
                            help='number of dev instance')
    
    


    args = arg_parser.parse_args()

    dataset = InstructData(args)
    dataset.create_data()