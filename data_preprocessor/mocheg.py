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
from nltk.tokenize import sent_tokenize
import glob
import csv

class InstructData:

    def __init__(self, args):
        self.train_dir = './raw_datasets/mocheg/mocheg/train/Corpus2.csv'
        self.val_dir = './raw_datasets/mocheg/mocheg/val/Corpus2.csv'
        self.train_img_dir = './raw_datasets/mocheg/mocheg/train/images'
        self.val_img_dir = './raw_datasets/mocheg/mocheg/val/images'
        self.vocab_dir = './data_preprocessor/label_mapping.csv'
        self.task_type = args.task_type
        self.out_data_dir = args.out_data_dir
        self.out_data_dir.mkdir(parents=True, exist_ok=True)
        self.meta_info_fn = 'meta.json'
        self.num_train = args.num_train
        self.num_val = args.num_val
        self.vocabs = {'supported':'yes','refuted':'no','NEI':'not sure'}
        self.gt_vocab = []
        with open(self.vocab_dir , 'r') as data:
            for line in csv.DictReader(data):
                if len(line['truthfulness'].strip()) > 0:
                    self.gt_vocab.append(line['truthfulness'])
            # print(self.gt_vocab)
        
        
    def create_multimodal_factual_checking_inst(self, save_path, input_dict, num_inst, split):
        count = 0
        with open(save_path,'w') as fout:
            for line in input_dict:
                claim_id = line['claim_id']
                claim = line['Claim']
                truthfulness = line['cleaned_truthfulness']
                evidences = line['ruling_outline']
                
                image_paths = []
                for img in glob.glob(f"./raw_datasets/mocheg/mocheg/{split}/images/{claim_id}-proof*"):
                    image_paths.append(img)
                if not len(image_paths)>0:
                    continue
                image_path = random.choice(image_paths)
                assert exists(image_path)
                evidences = sent_tokenize(evidences)
                # print('------------')
                # print(evidences)
                for vocab in self.gt_vocab:
                    if len(evidences) < 1:
                        break
                    if vocab in evidences[0]:
                        evidences.pop(0)
                    if len(evidences) < 1:
                        break
                        # print(vocab)
                        # print(evidences)
                    if vocab in evidences[-1]:
                        evidences.pop(-1)
                    if len(evidences) < 1:
                        break
                if len(evidences) < 1:
                    continue
                evidence = ' '.join(evidences)
                
                print(count)
                out_dict = {'unique_id': f"mocheg_{claim_id}",
                            'image_source': 'mocheg',
                            'task_name':  self.task_type,
                            'image_path': image_path,
                            'target_txt': self.vocabs[truthfulness],
                            'options': list(self.vocabs.values()),
                            'context': evidence,
                            'text': claim
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

        if self.task_type == 'multimodal_factual_checking':
            save_path = self.out_data_dir / 'valid.jsonl'
            input_dict = []
            with open(self.val_dir, 'r') as data:
                for line in csv.DictReader(data):
                    input_dict.append(line)
            random.shuffle(input_dict)
            self.create_multimodal_factual_checking_inst(save_path, input_dict, self.num_val, 'val')
            
            save_path = self.out_data_dir / 'train.jsonl'
            with open(self.train_dir, 'r') as data:
                for line in csv.DictReader(data):
                    input_dict.append(line)
            random.shuffle(input_dict)
            self.create_multimodal_factual_checking_inst(save_path, input_dict, self.num_train, 'train')
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