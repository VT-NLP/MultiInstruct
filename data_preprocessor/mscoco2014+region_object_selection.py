""" Given the region select the object from options"""

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
        
    def create_region_object_selection_inst(self, save_path, input_dict, num_inst, split):
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
                if len(v) < 4:
                    continue
                select_region_ids = random.sample(range(len(v)), 2)
                select_object_ids = random.sample(range(len(v)), random.randint(2,4))
                target_txt = set([])
                options = set([])
                
                for o_id in select_object_ids:
                    options.add(v[o_id][1])
                    in_region = False
                    for r_id in select_region_ids:
                        if v[o_id][1] == v[r_id][1]:
                            in_region = True
                            break
                    if in_region:
                        target_txt.add(v[o_id][1])
                region_obj = [ v[r_id][1] for r_id in select_region_ids]
                if split == 'valid':  
                    image_path = os.path.join(self.val_img_dir,f"COCO_val2014_{k:012d}.jpg")
                else:
                    image_path = os.path.join(self.train_img_dir,f"COCO_train2014_{k:012d}.jpg")
                assert exists(image_path)
                out_dict = {'unique_id': 'mscoco_detection2014_'+str(k),
                            'image_source': 'coco2014',
                            'task_name':  self.task_type,
                            'image_path': image_path,
                            'target_txt': ', '.join(list(target_txt)) if len(target_txt)>0 else 'None',
                            'region': [ v[_id][0] for _id in select_region_ids],
                            'options': list(options)+['None'],
                            'meta_data': {'region_obj':region_obj}
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

        if self.task_type == 'region_object_selection':
            save_path = self.out_data_dir / 'valid.jsonl'
            input_dict = json.load(open(self.val_dir, 'r'))
            self.create_region_object_selection_inst(save_path, input_dict, self.num_val, 'valid')
            
            save_path = self.out_data_dir / 'train.jsonl'
            input_dict = json.load(open(self.train_dir, 'r'))
            self.create_region_object_selection_inst(save_path, input_dict, self.num_train, 'train')
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