"""
Given two regions in the image decide if the object in the region are the same type
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
        self.vocabs = [['Yes','No'],['yes','no'],['same','different'],['yes, they are the same','no, they are not the same'],['same','not same'],['Yes, two objects are the same','No, two objects are different']]
        
    def create_object_match_inst(self, save_path, input_dict, num_inst, split):
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
            count = 0
            for k in random.sample(img2obj.keys(),len(img2obj)):
                v  = img2obj[k]
                if len(v) < 2:
                    continue
                select_ids = random.sample(range(len(v)), 2)
                vocabs = random.choice(self.vocabs)
                if v[select_ids[0]][1] == v[select_ids[1]][1]:
                    same_type = vocabs[0]
                else:
                    same_type = vocabs[1]
                if split == 'valid':  
                    image_path = os.path.join(self.val_img_dir,f"COCO_val2014_{k:012d}.jpg")
                else:
                    image_path = os.path.join(self.train_img_dir,f"COCO_train2014_{k:012d}.jpg")
                assert exists(image_path)
                out_dict = {'unique_id': 'mscoco_detection2014_'+str(k),
                            'image_source': 'coco2014',
                            'task_name':  self.task_type,
                            'image_path': image_path,
                            'target_txt': same_type,
                            'region': [v[select_ids[0]][0],v[select_ids[1]][0]],
                            'options': vocabs
                }
                fout.write(json.dumps(out_dict)+'\n')
                count += 1
                if count == num_inst:
                    break
        print(f'end loading coco2014 detection {split} data: ',save_path)
        
    def create_data(self):
        meta_data = {
            "originial_data_dir": str(self.train_dir)
        }
        
        with open(self.out_data_dir / self.meta_info_fn, 'w') as f:
            json.dump(meta_data, f)

        if self.task_type == 'object_match':
            save_path = self.out_data_dir / 'valid.jsonl'
            input_dict = json.load(open(self.val_dir, 'r'))
            self.create_object_match_inst(save_path, input_dict, self.num_val, 'valid')
            
            save_path = self.out_data_dir / 'train.jsonl'
            input_dict = json.load(open(self.train_dir, 'r'))
            self.create_object_match_inst(save_path, input_dict, self.num_train, 'train')
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