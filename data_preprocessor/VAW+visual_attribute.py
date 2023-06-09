
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
        self.data_dir = Path('./raw_datasets/VAW/vaw_dataset/data/')
        self.out_data_dir = args.out_data_dir
        self.out_data_dir.mkdir(parents=True, exist_ok=True)
        self.task_type = args.task_type
        self.meta_info_fn = 'meta.json'
        self.num_train = args.num_train
        self.num_valid = args.num_valid
        self.train_image_path = './raw_datasets/visual_genome/VG_100K'
        self.splits = ['train','val']

    def create_bool_q_data(self, save_fn, split, num_inst):
        inputs = []
        if split =='train':
            for i in ['train_part1.json','train_part2.json']:
                input_file = os.path.join(self.data_dir,i)
                input_data = json.load(open(input_file,'r'))
                inputs.extend(input_data)
        else:
            input_file = os.path.join(self.data_dir,f"val.json")
            inputs = json.load(open(input_file,'r'))
        random.shuffle(inputs)
        
        outputs = []
        count = 0
        for line in inputs:
            if len(line['positive_attributes'])<1 or  len(line['negative_attributes'])<1 :
                continue
            bbox = line['instance_bbox']
            bbox = [bbox[0],bbox[1],bbox[0]+bbox[2],bbox[1]+bbox[3]]
            target_txt = random.choice(line['positive_attributes'])
            negative_attributes = line['negative_attributes']
            image_path = os.path.join(self.train_image_path,f"{str(line['image_id'])}.jpg")
            try:
                assert exists(image_path)
            except:
                image_path = os.path.join(self.train_image_path+'_2',f"{str(line['image_id'])}.jpg")
                assert exists(image_path)
            unique_id = f"visual_attribute_{line['instance_id']}"
            out_dict = {'unique_id': unique_id,
                'task_name':  self.task_type,
                'image_path':image_path,
                'options': [target_txt]+negative_attributes,
                'target_txt': target_txt,
                'region': [bbox]
                }
            count += 1
            outputs.append(out_dict)
        # random.shuffle(outputs)
        with open(save_fn, 'w') as fout:
            count = 0
            for ex in outputs:
                fout.write(json.dumps(ex)+'\n')
                count += 1
                if count == num_inst:
                    return        
            
    def create_data(self):
    
        test_insts = []            
        if self.task_type == "visual_attribute":
            for split in self.splits:
                if split == 'train':
                    save_fn = self.out_data_dir / f'{split}.jsonl'
                    self.create_bool_q_data(save_fn,split,self.num_train)
                elif split =='val':
                    save_fn = self.out_data_dir / f'valid.jsonl'
                    self.create_bool_q_data(save_fn,'dev',self.num_valid)
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
    arg_parser.add_argument('--num_train', type=int,
                            default=300,
                            help='number of testing instance')
    arg_parser.add_argument('--num_valid', type=int,
                            default=300,
                            help='number of testing instance')
    args = arg_parser.parse_args()

    dataset = InstructData(args)
    dataset.create_data()