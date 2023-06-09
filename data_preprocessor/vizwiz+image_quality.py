"""
image quality
"""

from argparse import ArgumentParser
import json
from pathlib import Path
# import pdb
# import pandas as pd
import os
import random
from os.path import exists

class InstructData:

    def __init__(self, args):
        self.train_dir = './raw_datasets/vizwiz_image_quality/train.json'
        self.val_dir = './raw_datasets/vizwiz_image_quality/val.json'
        self.train_img_dir = './raw_datasets/vizwiz_image_quality/train'
        self.val_img_dir = './raw_datasets/vizwiz_image_quality/val'
        self.task_type = args.task_type
        self.out_data_dir = args.out_data_dir
        self.out_data_dir.mkdir(parents=True, exist_ok=True)
        self.meta_info_fn = 'meta.json'
        self.num_train = args.num_train
        self.num_val = args.num_val
        self.vocabs = { 'NON':'no flaws',
                        'BLR':'blur',
                        'BRT': 'too bright',
                        'DRK': 'too dark',
                        'FRM': 'bad framing',
                        'OBS': 'obscured',
                        'ROT': 'rotation',
                        'OTH': 'other'
                       }
        
    def create_image_quality_inst(self, save_path, input_dict, num_inst, split):
        count = 0
        with open(save_path,'w') as fout:
            for line in input_dict:
                flaws = line['flaws']
                flaw_num = 0
                for k, v in flaws.items():
                    if v > flaw_num:
                        flaw = k
                        flaw_num = v
                
                if split == 'valid':  
                    image_path = f"{self.val_img_dir}/{line['image']}"
                else:
                    image_path = f"{self.train_img_dir}/{line['image']}"
                assert exists(image_path)
                out_dict = {'unique_id': f"vizwiz_image_quality_{count+1}",
                            'image_source': 'vizwiz',
                            'task_name':  self.task_type,
                            'image_path': image_path,
                            'target_txt': self.vocabs[flaw],
                            'options': list(self.vocabs.values())
                }
                fout.write(json.dumps(out_dict)+'\n')
                count += 1
                if count == num_inst:
                    break
        print(f'end loading {split} data: ',save_path)
        
    def create_data(self):
        meta_data = {
            "originial_data_dir": str(self.train_dir)
        }
        
        with open(self.out_data_dir / self.meta_info_fn, 'w') as f:
            json.dump(meta_data, f)

        if self.task_type == 'image_quality':
            save_path = self.out_data_dir / 'valid.jsonl'
            input_dict = json.load(open(self.val_dir, 'r'))
            self.create_image_quality_inst(save_path, input_dict, self.num_val, 'valid')
            
            save_path = self.out_data_dir / 'train.jsonl'
            input_dict = json.load(open(self.train_dir, 'r'))
            self.create_image_quality_inst(save_path, input_dict, self.num_train, 'train')
        else:
            raise NotImplementedError

if __name__ == '__main__':
    arg_parser = ArgumentParser(
        'Create dataset for Hateful Memes')
    arg_parser.add_argument('--out_data_dir', type=Path,
                            help='Path to the output data folder')
    arg_parser.add_argument('--task_type', type=str,
                            help='Specify the type of task to create dataset')
    arg_parser.add_argument('--num_train', type=int,
                            help='number of training instance')
    arg_parser.add_argument('--num_val', type=int,
                            help='number of dev instance')
    
    args = arg_parser.parse_args()

    dataset = InstructData(args)
    dataset.create_data()