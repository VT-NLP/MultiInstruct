
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
        self.data_dir = Path('./raw_datasets/visual_spatial_reasoning//visual-spatial-reasoning/data')
        self.out_data_dir = args.out_data_dir
        self.out_data_dir.mkdir(parents=True, exist_ok=True)
        self.task_type = args.task_type
        self.meta_info_fn = 'meta.json'
        # self.num_train = args.num_train
        # self.num_valid = args.num_valid
        self.train_image_path = f"{self.data_dir}/trainval2017"
        # self.splits = ['train','val']

    def create_bool_q_data(self, save_fn):
        
        input_file = os.path.join(self.data_dir,f"splits/zeroshot/test.jsonl")
        inputs = []
        with open(input_file,'r') as fin:
            for line in fin:
                line = json.loads(line)
                inputs.append(line)
        random.shuffle(inputs)
        outputs = []
        count = 0
        for line in inputs:
            text = line['caption']
            target_txt = 'yes' if line['label'] == 1 else 'no'
            image_path = os.path.join(self.train_image_path,f"{line['image']}")
            exists(image_path)
            unique_id = f"visual_spatial_reasoning_test_{count}"
            out_dict = {'unique_id': unique_id,
                'task_name':  self.task_type,
                'image_path':image_path,
                'text': text,
                'target_txt': target_txt,
                'options': ['yes','no']
                }
            count += 1
            outputs.append(out_dict)
        # random.shuffle(outputs)
        with open(save_fn, 'w') as fout:
            for ex in outputs:
                fout.write(json.dumps(ex)+'\n')     
            
    def create_data(self):
    
        test_insts = []            
        if self.task_type == "visual_spatial_reasoning":
            save_fn = self.out_data_dir / f'test.jsonl'
            self.create_bool_q_data(save_fn)
        else:
            raise NotImplemented

if __name__ == '__main__':
    arg_parser = ArgumentParser(
        'Create dataset for Text VQA')
    arg_parser.add_argument('--out_data_dir', type=Path,
                            default='unseen_data/text_vqa',
                            help='Path to the output data folder')
    arg_parser.add_argument('--task_type', type=str,
                            default='',
                            help='Specify the type of task to create dataset')
    args = arg_parser.parse_args()

    dataset = InstructData(args)
    dataset.create_data()