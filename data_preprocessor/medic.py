"""what is the text in the image"""

from argparse import ArgumentParser
import csv
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
        self.data_dir = Path('./raw_datasets/MEDIC')
        self.out_data_dir = args.out_data_dir 
        self.out_data_dir.mkdir(parents=True, exist_ok=True)
        self.meta_info_fn = 'meta.json'
        self.num_test = args.num_test
        self.image_path = self.data_dir
        self.anno_fn = 'MEDIC_test.tsv'
        self.options = {
            'damage_severity': ['severe damage', 'mild damage', 'little or none damage'],
            'informative': ['informative', 'not informative'],
            'disaster_types': ['earthquake', 'fire', 'flood', 'hurricane', 'landslide', 'not disaster', 'other disaster']
        }
        self.task_categories = ['damage_severity', 'informative', 'disaster_types']
                    
    # def create_data(self):
    #     meta_data = {
    #         "originial_data_dir": str(self.data_dir)
    #     }
        
    #     with open(self.out_data_dir / self.meta_info_fn, 'w') as f:
    #         json.dump(meta_data, f)
            
    #     save_fn = self.out_data_dir / f'test.jsonl'
    #     with open(self.data_dir / self.anno_fn, 'r') as tsv_file:
    #         with open(save_fn, 'w') as fout:
    #             reader = csv.DictReader(tsv_file, delimiter='\t')
    #             count = 0
    #             for row in reader:
                    
    #                 image_path = str(self.data_dir / row['image_path'])
    #                 assert exists(image_path)
    #                 image_source = 'medic'
    #                 unique_id = f"medic_test_{count}"
                    
    #                 for task_cat in self.task_categories:
    #                     target_txt = ' '.join(row[task_cat].split('_')).lower()
    #                     if task_cat == 'damage_severity':
    #                         target_txt = f'{target_txt} damage'
    #                     if target_txt not in self.options:
    #                         raise ValueError(f'target {target_txt} not in options: {self.options}')
            
    #                     out_dict = {'unique_id': f"{unique_id}_{task_cat}",
    #                                 'image_source': image_source,
    #                                 'task_name':  f'medic_{task_cat}',
    #                                 'image_path': image_path,
    #                                 'target_txt': target_txt,
    #                                 'options': self.options,
    #                                 'meta_data': {'task_cat': task_cat}
    #                     }
    #                     fout.write(json.dumps(out_dict)+'\n')
    #                 count+=1
    #                 if count == self.num_test:
    #                     return
    
    def create_data(self):
        
        
        option_lst = sum(list(self.options.values()), [])
        
        save_fn = {}
        outputs = {}
        for task_cat in self.task_categories:
            save_fn[task_cat] = self.out_data_dir / f'medic_{task_cat}' / f'test.jsonl'
            save_fn[task_cat].parent.mkdir(parents=True, exist_ok=True)
            outputs[task_cat] = []
        
        with open(self.data_dir / self.anno_fn, 'r') as tsv_file:
            # with open(save_fn, 'w') as fout:
                reader = csv.DictReader(tsv_file, delimiter='\t')
                count = 0
                for row in reader:
                    
                    image_path = str(self.data_dir / row['image_path'])
                    assert exists(image_path)
                    image_source = 'medic'
                    unique_id = f"medic_test_{count}"
                    
                    for task_cat in self.task_categories:
                        target_txt = ' '.join(row[task_cat].split('_')).lower()
                        if task_cat == 'damage_severity':
                            target_txt = f'{target_txt} damage'
                        if target_txt not in option_lst:
                            raise ValueError(f'target {target_txt} not in options: {option_lst}')
            
                        out_dict = {'unique_id': f"{unique_id}_{task_cat}",
                                    'image_source': image_source,
                                    'task_name':  f'medic_{task_cat}',
                                    'image_path': image_path,
                                    'target_txt': target_txt,
                                    'options': option_lst,
                                    'meta_data': {'task_cat': task_cat, 'options': self.options[task_cat]}
                        }
                        outputs[task_cat].append(out_dict)
                        # fout.write(json.dumps(out_dict)+'\n')
                    count+=1
                    if count == self.num_test:
                        return
                    
        for task_cat in self.task_categories:
            with open(save_fn[task_cat], 'w') as fout:
                for ex in outputs[task_cat]:
                    fout.write(json.dumps(ex)+'\n')
                

if __name__ == '__main__':
    arg_parser = ArgumentParser(
        'Create dataset for MEDIC')
    arg_parser.add_argument('--out_data_dir', type=Path,
                            default='unseen_data/',
                            help='Path to the output data folder')
    arg_parser.add_argument('--num_test', type=int,
                            help='number of testing instance')
    args = arg_parser.parse_args()

    dataset = InstructData(args)
    dataset.create_data()