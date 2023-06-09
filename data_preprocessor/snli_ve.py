"""decide if the image content can support the text"""
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
        self.split = 'test'
        self.data_dir = './raw_datasets/SNLI-VE/SNLI-VE/data'
        self.out_data_dir = args.out_data_dir 
        self.out_data_dir.mkdir(parents=True, exist_ok=True)
        self.meta_info_fn = 'meta.json'
        self.num_test = args.num_test
        self.image_path = os.path.join(self.data_dir, 'Flickr30K', 'images')
        self.task_type = args.task_type
        self.vocabs = {'neutral':'not sure','entailment':'yes','contradiction':'no'}

    def create_visual_nli_inst(self,save_fn):
        with open(os.path.join(self.data_dir, f"snli_ve_{self.split}.jsonl"), 'r') as fin:
            with open(save_fn, 'w') as fout:
                count = 0
                for line in fin:
                    line = json.loads(line)
                    unique_id = f"Flickr30K_ID_{line['Flickr30K_ID']}_{line['pairID']}"
                    image_source = 'Flickr30K_ID'
                    image_path = os.path.join(self.image_path,f"{line['Flickr30K_ID']}.jpg")
                    assert exists(image_path)
                    out_dict = {'unique_id': unique_id,
                                'image_source': image_source,
                                'task_name':  self.task_type,
                                'image_path': image_path,
                                'target_txt': self.vocabs[line['gold_label']],
                                'text': line['sentence2'],
                                'options':list(self.vocabs.values()),
                                'meta_data': {'caption':line['sentence1']}
                    }
                    fout.write(json.dumps(out_dict)+'\n')
                    count+=1
                    if count == self.num_test:
                        return
                   
    def create_data(self):

        meta_data = {
            "originial_data_dir": self.image_path
        }
        
        with open(self.out_data_dir / self.meta_info_fn, 'w') as f:
            json.dump(meta_data, f)

        save_fn = self.out_data_dir / f'{self.split}.jsonl'
        
        if self.task_type == 'visual_nli':
            self.create_visual_nli_inst(save_fn)
        else:
            raise NotImplementedError


if __name__ == '__main__':
    arg_parser = ArgumentParser(
        'Create dataset for Visual Entailment')
    arg_parser.add_argument('--out_data_dir', type=Path,
                            help='Path to the output data folder')
    arg_parser.add_argument('--task_type', type=str,
                            help='Specify the type of task to create dataset')
    arg_parser.add_argument('--num_test', type=int,
                            help='number of testing instance')
    args = arg_parser.parse_args()

    dataset = InstructData(args)
    dataset.create_data()
