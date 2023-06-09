"""what is the text in the image"""

from argparse import ArgumentParser
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
        self.data_dir = './raw_datasets/hateful_memes/'
        self.out_data_dir = args.out_data_dir 
        self.out_data_dir.mkdir(parents=True, exist_ok=True)
        self.meta_info_fn = 'meta.json'
        self.num_test = args.num_test
        self.image_path = self.data_dir
        self.task_type = args.task_type
        # self.splits = ['test_seen', 'test_unseen']
        self.splits = ['test_unseen']
        
    def create_visual_text_extraction_inst(self, save_fn, split):
        with open(os.path.join(self.data_dir , f'{split}.jsonl'), 'r') as fin:
            with open(save_fn, 'w') as fout:
                count = 0
                for line in fin:
                    line = json.loads(line)
                    unique_id = f"hateful_memes_{split}_{line['id']}"
                    image_path = os.path.join(self.data_dir,line['img'])
                    assert exists(image_path)
                    text = line['text']
                    label = "Yes" if line['label'] == 1 else 'No'
                    image_source = 'hateful_memes'
                    out_dict = {'unique_id': unique_id,
                                'image_source': image_source,
                                'task_name':  self.task_type,
                                'image_path': image_path,
                                'target_txt': text,
                                'meta_data': {'label':label}
                    }
                    fout.write(json.dumps(out_dict)+'\n')
                    count+=1
                    if count == self.num_test:
                        return
                    
    def create_hateful_content_detection_inst(self, save_fn, split):
        with open(os.path.join(self.data_dir , f'{split}.jsonl'), 'r') as fin:
            with open(save_fn, 'w') as fout:
                count = 0
                for line in fin:
                    line = json.loads(line)
                    unique_id = f"hateful_memes_{split}_{line['id']}"
                    image_path = os.path.join(self.data_dir,line['img'])
                    assert exists(image_path)
                    text = line['text']
                    label = "Yes" if line['label'] == 1 else 'No'
                    image_source = 'hateful_memes'
                    out_dict = {'unique_id': unique_id,
                                'image_source': image_source,
                                'task_name':  self.task_type,
                                'image_path': image_path,
                                'target_txt': label,
                                'options': ['Yes','No'],
                                'meta_data': {'text':text}
                    }
                    fout.write(json.dumps(out_dict)+'\n')
                    count+=1
                    if count == self.num_test:
                        return
            
    
    def create_data(self):

        meta_data = {
            "originial_data_dir": str(self.data_dir)
        }
        
        with open(self.out_data_dir / self.meta_info_fn, 'w') as f:
            json.dump(meta_data, f)

        for split in self.splits:
            save_fn = self.out_data_dir / f'test.jsonl'
            
            if self.task_type == 'visual_text_extraction':
                self.create_visual_text_extraction_inst(save_fn,split)
            
            elif self.task_type == 'hateful_content_detection':
                self.create_hateful_content_detection_inst(save_fn,split)
            
            else:
                raise NotImplementedError


if __name__ == '__main__':
    arg_parser = ArgumentParser(
        'Create dataset for Hateful Memes')
    arg_parser.add_argument('--out_data_dir', type=Path,
                            help='Path to the output data folder')
    arg_parser.add_argument('--task_type', type=str,
                            help='Specify the type of task to create dataset')
    arg_parser.add_argument('--num_test', type=int,
                            help='number of testing instance')
    args = arg_parser.parse_args()

    dataset = InstructData(args)
    dataset.create_data()