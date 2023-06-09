"""Answer the question by selecting the region"""
"""{"sentence":"There is atleast one grey box with exactly three objects","label":"false","identifier":"1731-3","directory":"10","evals":{"r0":"false"},"structured_rep":[[{"y_loc":3,"size":20,"type":"square","x_loc":76,"color":"#0099ff"}],[{"y_loc":73,"size":20,"type":"circle","x_loc":2,"color":"Black"},{"y_loc":87,"size":10,"type":"square","x_loc":87,"color":"Black"},{"y_loc":43,"size":20,"type":"triangle","x_loc":72,"color":"Yellow"},{"y_loc":90,"size":10,"type":"triangle","x_loc":57,"color":"Black"},{"y_loc":4,"size":10,"type":"circle","x_loc":23,"color":"Yellow"}],[{"y_loc":61,"size":20,"type":"circle","x_loc":13,"color":"Yellow"},{"y_loc":35,"size":30,"type":"circle","x_loc":56,"color":"#0099ff"},{"y_loc":77,"size":20,"type":"circle","x_loc":55,"color":"Yellow"},{"y_loc":2,"size":30,"type":"circle","x_loc":70,"color":"Yellow"}]]}
"""
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
        self.data_dir = Path('./raw_datasets/nlvr/nlvr/nlvr')
        self.out_data_dir = args.out_data_dir
        self.out_data_dir.mkdir(parents=True, exist_ok=True)
        self.task_type = args.task_type
        self.meta_info_fn = 'meta.json'

    def create_bool_q_data(self, save_fn):
        input_file = os.path.join(self.data_dir,f"test/test.json")
        inputs = []
        with open(input_file,'r') as fin:
            for line in fin:
                line = json.loads(line)
                inputs.append(line)
        random.shuffle(inputs)
        
        outputs = []
        for line in inputs:
            text = line['sentence']
            target_txt = 'yes' if line['label'] == 'true' else 'no'
            image_path = os.path.join(self.data_dir,f"test/images/{line['directory']}/test-{line['identifier']}-{random.choice(['0','1','2','3','4','5'])}.png")
            assert exists(image_path)
            unique_id = f"nlvr_{line['directory']}_{line['identifier']}"
            out_dict = {'unique_id': unique_id,
                'task_name':  self.task_type,
                'image_path':image_path,
                'text': text,
                'target_txt': target_txt,
                'options': ['yes','no']
                }
            outputs.append(out_dict)
        with open(save_fn, 'w') as fout:
            count = 0
            for ex in outputs:
                fout.write(json.dumps(ex)+'\n')    
            
    def create_data(self):
    
        test_insts = []            
        if self.task_type == "natural_language_visual_reasoning":
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