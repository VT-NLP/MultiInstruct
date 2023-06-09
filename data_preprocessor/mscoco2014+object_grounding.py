"""  Given 1 or 2 or 3 regions and predict what objects are in the regions
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
        self.train_dir = './raw_datasets/MSCOCO2014/annotations/instances_train2014.json'
        self.val_dir = './raw_datasets/MSCOCO2014/annotations/instances_val2014.json'
        self.train_img_dir = './raw_datasets/MSCOCO2014/train2014'
        self.val_img_dir = './raw_datasets/MSCOCO2014/val2014'
        self.task_type = args.task_type
        self.out_data_dir = args.out_data_dir
        self.out_data_dir.mkdir(parents=True, exist_ok=True)
        self.meta_info_fn = 'meta.json'
        self.num_train = args.num_train
        self.num_val = args.num_val
        
    def create_object_grounding_inst(self, save_path, input_dict, num_inst, split):
        with open(save_path,'w') as fout:
            category_list = input_dict['categories']
            category_dict = {}
            options = []
            for cat in category_list:
                category_dict[str(cat['id'])] = cat['name']
                options.append(cat['name'])
            # images = input_dict['images']
            annotations = input_dict['annotations']
            img2obj = {}
            for line in annotations:
                bbox = [line['bbox'][0],line['bbox'][1], line['bbox'][0]+line['bbox'][2], line['bbox'][1]+line['bbox'][3]]
                if line['image_id'] in img2obj:
                    img2obj[line['image_id']].append([bbox,category_dict[str(line['category_id'])]])
                else:
                    img2obj[line['image_id']] = [[bbox,category_dict[str(line['category_id'])]]]
            print(f'total num of {split} {len(annotations)}, span num of {split} {num_inst}')
            if num_inst == -1 or len(img2obj) < num_inst:
                input_keys = img2obj.keys()
            else:
                input_keys = random.sample(img2obj.keys(),num_inst)
            for k in input_keys:
                v  = img2obj[k]
                select_index = random.sample(range(len(v)), 1)
                if split == 'valid':  
                    image_path = os.path.join(self.val_img_dir,f"COCO_val2014_{k:012d}.jpg")
                else:
                    image_path = os.path.join(self.train_img_dir,f"COCO_train2014_{k:012d}.jpg")
                assert exists(image_path)
                out_dict = {'unique_id': 'mscoco_detection2014_'+str(k),
                            'image_source': 'coco2014',
                            'task_name':  self.task_type,
                            'image_path': image_path,
                            'target_txt': ', '.join([v[idx][1] for idx in select_index]),
                            'region': [v[idx][0] for idx in select_index]
                }
                fout.write(json.dumps(out_dict)+'\n')
        print(f'end loading coco2014 detection {split} data: ',save_path)
        
    def create_data(self):
        meta_data = {
            "originial_data_dir": str(self.train_dir)
        }
        
        with open(self.out_data_dir / self.meta_info_fn, 'w') as f:
            json.dump(meta_data, f)

        if self.task_type == 'object_grounding':
            save_path = self.out_data_dir / 'valid.jsonl'
            input_dict = json.load(open(self.val_dir, 'r'))
            self.create_object_grounding_inst(save_path, input_dict, self.num_val, 'valid')
            
            save_path = self.out_data_dir / 'train.jsonl'
            input_dict = json.load(open(self.train_dir, 'r'))
            self.create_object_grounding_inst(save_path, input_dict, self.num_train, 'train')
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