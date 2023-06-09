"""Answer the question by selecting the region"""
from argparse import ArgumentParser
from ast import Not
import json
from pathlib import Path
# import pdb
# import pandas as pd
import os
import random
from os.path import exists
import pdb

class InstructData:

    def __init__(self, args):
        self.data_dir = Path('./raw_datasets/visual7w/')
        self.out_data_dir = args.out_data_dir
        self.out_data_dir.mkdir(parents=True, exist_ok=True)
        self.task_type = args.task_type
        self.meta_info_fn = 'meta.json'
        self.num_train = args.num_train
        self.num_valid = args.num_valid
        self.image_path = f"{self.data_dir}/images"
        self.splits = ['train','val']

    def create_telling_grounded_VQA_data(self, save_fn, split, num_inst):
        telling = json.load(open(f'{self.data_dir}/dataset_v7w_telling.json','r'))
        
        all_qa_pairs = []
        for qa_pairs in telling['images']:
            if qa_pairs['split'] == split:
                img_file = qa_pairs['filename']
                img_id = qa_pairs['image_id']
                for qa_pair in qa_pairs['qa_pairs']:
                    unique_id = f"{img_id}_{qa_pair['qa_id']}"
                    answer = [qa_pair['answer']]
                    choices = qa_pair['multiple_choices'] + answer
                    random.shuffle(choices)
                    image_source = 'Visual7W'
                    image_path = f"{self.image_path}/{img_file}"
                    assert exists(image_path)
                    out_dict = {'unique_id': unique_id,
                        'image_source': image_source,
                        'task_name':  self.task_type,
                        'image_path': image_path,
                        'question': qa_pair['question'],
                        'options': choices,
                        'target_txt': answer[0],
                        'meta_data': {'question_type': qa_pair['type']}
                        }
                    all_qa_pairs.append(out_dict)
        random.shuffle(all_qa_pairs)
        with open(save_fn, 'w') as fout:
            count = 0
            for ex in all_qa_pairs:
                fout.write(json.dumps(ex)+'\n')
                count += 1
                if count == num_inst:
                    return
            
    def create_data(self):
    
        test_insts = []            
        if self.task_type == "VQA":
            for split in self.splits:
                if split == 'train':
                    save_fn = self.out_data_dir / f'{split}.jsonl'
                    self.create_telling_grounded_VQA_data(save_fn,split,self.num_train)
                elif split =='val':
                    save_fn = self.out_data_dir / f'valid.jsonl'
                    self.create_telling_grounded_VQA_data(save_fn,split,self.num_valid)
            
        else:
            raise NotImplemented

if __name__ == '__main__':
    arg_parser = ArgumentParser(
        'Create dataset for Text VQA')
    arg_parser.add_argument('--out_data_dir', type=Path,
                            default='unseen_data/text_vqa',
                            help='Path to the output data folder')
    arg_parser.add_argument('--task_type', type=str,
                            default='ocr',
                            choices=['VQA'],
                            help='Specify the type of task to create dataset')
    arg_parser.add_argument('--num_train', type=int,
                            default=300,
                            help='number of testing instance')
    arg_parser.add_argument('--num_valid', type=int,
                            default=300,
                            help='number of testing instance')
    args = arg_parser.parse_args()

    dataset = InstructData(args)
    dataset.create_data()