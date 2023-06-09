
from argparse import ArgumentParser
import json
from pathlib import Path
import pdb
import os
import random
from os.path import exists
import sys
sys.path.append('..')
from tqdm import tqdm

# from image_code_utils import process_code

class InstructData:

    def __init__(self, args):
        self.data_dir = Path('./raw_datasets/wikihow/')
        self.data_split_fn = 'data_preprocessor/wikihow_split.json'
        self.img_dir = Path('./raw_datasets/wikihow/data')
        self.task_type = args.task_type
        self.out_data_dir = args.out_data_dir
        self.out_data_dir.mkdir(parents=True, exist_ok=True)
        self.wikihow_fn = 'wikihow.json'
        self.num_train = args.num_train
        self.num_val = args.num_val
    
    def create_nxt_step_data(self):
        
        data_dict = {
            'train': [],
            'val': []
        }
        
        with open(self.data_split_fn, 'r') as f:
            data_split_dict = json.load(f)
        
        for split, dir_list in tqdm(data_split_dict.items(), desc='processing data'):
            if split == 'test':
                continue
            for dir_name in dir_list:
                with open(self.data_dir/ 'data' / dir_name / self.wikihow_fn, 'r') as f:
                    data = json.load(f)
                if 'methods' in data:
                    for method in data['methods']:
                        history_context = []
                        for i in range(len(method['steps']) - 1):
                            if method['steps'][i]['img']:
                                img_fn = str(self.img_dir / Path(dir_name) / 'image' / method['steps'][i]['img'])
                                exists(img_fn)
                                cur_text = method['steps'][i]['headline']
                                nxt_text = method['steps'][i+1]['headline']
                                out_dict = {'unique_id': f'wikihow_{self.task_type}_{dir_name}',
                                            'image_source': 'wikihow',
                                            'task_name':  self.task_type,
                                            'image_path': img_fn,
                                            'text': cur_text,
                                            'context': history_context,
                                            'target_txt': nxt_text,
                                            'meta_data': {'method':method['name']},
                                            }
                                history_context.append(cur_text)
                                data_dict[split].append(out_dict)
        
        return data_dict    
    
    def create_prev_step_data(self):
        
        data_dict = {
            'train': [],
            'val': []
        }
        
        with open(self.data_split_fn, 'r') as f:
            data_split_dict = json.load(f)
        
        for split, dir_list in tqdm(data_split_dict.items(), desc='processing data'):
            if split == 'test':
                continue
            for dir_name in dir_list:
                with open(self.data_dir/ 'data' / dir_name / self.wikihow_fn, 'r') as f:
                    data = json.load(f)
                if 'methods' in data:
                    for method in data['methods']:
                        for i in range(1, len(method['steps'])):
                            if method['steps'][i]['img']:
                                img_fn = str(self.img_dir / Path(dir_name) / 'image' / method['steps'][i]['img'])
                                exists(img_fn)
                                cur_text = method['steps'][i]['headline']
                                nxt_text = method['steps'][i-1]['headline']
                                out_dict = {'unique_id': f'wikihow_{self.task_type}_{dir_name}',
                                            'image_source': 'wikihow',
                                            'task_name':  self.task_type,
                                            'image_path': img_fn,
                                            'text': cur_text,
                                            'target_txt': nxt_text,
                                            'meta_data': {'method': method['name']},
                                            }
                                data_dict[split].append(out_dict)
        
        return data_dict    


    def create_img_txt_step_order_data(self):
        
        data_dict = {
            'train': [],
            'val': []
        }
        
        with open(self.data_split_fn, 'r') as f:
            data_split_dict = json.load(f)
        
        for split, dir_list in data_split_dict.items():
            if split == 'test':
                continue
            for dir_name in tqdm(dir_list, desc=f'processing {split} data'):
                with open(self.data_dir/ 'data' / dir_name / self.wikihow_fn, 'r') as f:
                    data = json.load(f)
                if 'methods' in data:
                    for method in data['methods']:
                        for i in range(1, len(method['steps']) - 1):
                            if not method['steps'][i]['img']:
                                    continue
                            img_fn = str(self.img_dir / Path(dir_name) / 'image' / method['steps'][i]['img'])
                            exists(img_fn)
                            if random.uniform(0, 1) > 0.5: # next step
                                target_txt = 'next'
                                text = method['steps'][i+1]['headline']
                            else: # prev step
                                target_txt = 'previous'
                                text = method['steps'][i-1]['headline']
                                
                            out_dict = {'unique_id': f'wikihow_{self.task_type}_{dir_name}',
                                        'image_source': 'wikihow',
                                        'task_name':  self.task_type,
                                        'image_path': img_fn,
                                        'text': text,
                                        'target_txt': target_txt,
                                        'meta_data': {'method':method['name']},
                            }
                            data_dict[split].append(out_dict)
        return data_dict    
    
    def create_txt_img_step_order_data(self):
        
        data_dict = {
            'train': [],
            'val': []
        }
        
        with open(self.data_split_fn, 'r') as f:
            data_split_dict = json.load(f)
        
        for split, dir_list in data_split_dict.items():
            if split == 'test':
                continue
            for dir_name in tqdm(dir_list, desc=f'processing {split} data'):
                with open(self.data_dir/ 'data' / dir_name / self.wikihow_fn, 'r') as f:
                    data = json.load(f)
                if 'methods' in data:
                    for method in data['methods']:
                        # indices = random.sample(range(len(method['steps'])), 2)
                        # if method['steps'][indices[1]]['img']:
                        for i in range(1, len(method['steps']) - 1):
                            if random.uniform(0, 1) > 0.5: # next step
                                if not method['steps'][i+1]['img']:
                                    continue
                                target_txt = 'next'
                                img_fn = str(self.img_dir / Path(dir_name) / 'image' / method['steps'][i+1]['img'])
                                exists(img_fn)
                                text = method['steps'][i]['headline']
                            
                            else: # prev step
                                if not method['steps'][i-1]['img']:
                                    continue
                                target_txt = 'previous'
                                img_fn = str(self.img_dir / Path(dir_name) / 'image' / method['steps'][i-1]['img'])
                                exists(img_fn)
                                text = method['steps'][i]['headline']
                                
                            out_dict = {'unique_id': f'wikihow_{self.task_type}_{dir_name}',
                                        'image_source': 'wikihow',
                                        'task_name':  self.task_type,
                                        'image_path': img_fn,
                                        'text': text,
                                        'target_txt': target_txt,
                                        'meta_data': {'method':method['name']},
                            }
                            data_dict[split].append(out_dict)
    
        return data_dict    
    
    def create_immediate_nxt_step_data(self):
        
        data_dict = {
            'train': [],
            'val': []
        }
        
        with open(self.data_split_fn, 'r') as f:
            data_split_dict = json.load(f)
        
        for split, dir_list in data_split_dict.items():
            if split == 'test':
                continue
            for dir_name in tqdm(dir_list, desc=f'processing {split} data'):
                with open(self.data_dir/ 'data' / dir_name / self.wikihow_fn, 'r') as f:
                    data = json.load(f)
                if 'methods' in data:
                    for method in data['methods']:
                        for i in range(len(method['steps']) - 1):
                            if method['steps'][i]['img']:
                                img_fn = str(self.img_dir / Path(dir_name) / 'image' / method['steps'][i]['img'])
                                exists(img_fn)
                                nxt_text = method['steps'][i+1]['headline']
                                options = []
                                selections = list(range(0,i)) + list(range(i+2, len(method['steps'])))
                                sample_num = int(len(selections) / 2) if len(selections) / 2 >= 2 else min(2, len(selections))
                                options = random.sample(selections, sample_num)
                                if options:
                                    options.append(i+1)
                                    options = [method['steps'][idx]['headline'] for idx in options]
                                    out_dict = {'unique_id': f'wikihow_{self.task_type}_{dir_name}',
                                                'image_source': 'wikihow',
                                                'task_name':  self.task_type,
                                                'image_path': img_fn,
                                                'options': options,
                                                'target_txt': nxt_text,
                                                'meta_data': {'method':method['name']},
                                                }
                                    data_dict[split].append(out_dict)
        
        return data_dict    
    
    
    def create_data(self):

        if self.task_type == 'wikihow_next_step':
            data_dict = self.create_nxt_step_data()
        elif self.task_type == 'wikihow_text_image_step_order':
            # given the text, specify if the image is the previous or next step
            data_dict = self.create_txt_img_step_order_data()
        elif self.task_type == 'wikihow_image_text_step_order':
            # given the image, specify if the text is the previous or next step
            data_dict = self.create_img_txt_step_order_data()
        elif self.task_type == 'wikihow_immediate_next_step_selection':
            # given image, which “text” is the immediate next step
            data_dict = self.create_immediate_nxt_step_data()
        else:
            raise NotImplemented
        
        for split, data in data_dict.items():
            if split == 'train':
                data = data[:self.num_train]
            elif split == 'val':
                data = data[:self.num_val]
            print(f'{self.task_type}[{split}] = {len(data)}')
            
            if split == 'train':
                save_path = self.out_data_dir / f'{split}.jsonl'
            elif split == 'val':
                save_path = self.out_data_dir / f'valid.jsonl'
            
            with open(save_path,'w') as fout:
                for ex in data:
                    fout.write(json.dumps(ex)+'\n')
           

if __name__ == '__main__':
    arg_parser = ArgumentParser(
        'Create dataset for Wikihow')
    arg_parser.add_argument('--out_data_dir', type=Path,
                            default='instruct_data',
                            help='Path to the output data folder')
    arg_parser.add_argument('--task_type', type=str,
                            default='wikihow_next_step',
                            choices=['wikihow_next_step', 'wikihow_text_image_step_order', 'wikihow_image_text_step_order', 'wikihow_immediate_next_step_selection'],
                            help='Specify the type of task to create dataset')
    arg_parser.add_argument('--num_train', type=int,
                            help='number of training instance')
    arg_parser.add_argument('--num_val', type=int,
                            help='number of dev instance')
    


    args = arg_parser.parse_args()

    dataset = InstructData(args)
    dataset.create_data()