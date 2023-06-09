"""Answer the question by selecting the region"""
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
        self.data_dir = Path('./raw_datasets/visual7w/')
        self.out_data_dir = args.out_data_dir
        self.out_data_dir.mkdir(parents=True, exist_ok=True)
        self.task_type = args.task_type
        self.meta_info_fn = 'meta.json'
        self.num_test = args.num_test
        self.image_path = f"{self.data_dir}/images"
        self.split = 'test'
 
    def create_grounded_VQA_data(self, save_fn):
        pointing = json.load(open(f'{self.data_dir}/dataset_v7w_pointing.json','r'))
        bbox = {}
        for box in pointing['boxes']:
            obj = box['name']
            box_id = box['box_id']
            assert not box_id in bbox
            x = box['x'] + 1 if box['x'] == -1 else box['x']
            y = box['y'] + 1 if box['y'] == -1 else box['y']
            width = box['width']
            height = box['height']
            assert not box_id in bbox
            bbox[box_id] = {'obj':obj, 'bbox':[x,y,x+width,y+height]}
            
        skip = 0
        all_qa_pairs = []
        for qa_pairs in pointing['images']:
            if qa_pairs['split'] == 'test':
                img_file = qa_pairs['filename']
                img_id = qa_pairs['image_id']
                for qa_pair in qa_pairs['qa_pairs']:
                    unique_id = f"visual7w_pointing_{img_id}_{qa_pair['qa_id']}"
                    answer = bbox[qa_pair['answer']]['bbox']
                    choices = [ bbox[ch]['bbox'] for ch in qa_pair['multiple_choices']] + [answer]
                    # if answer[0][0] < 0 or answer[0][1] < 0 or answer[0][2] < 0 or answer[0][3] < 0:
                    #     skip+=1
                    #     continue
                    random.shuffle(choices)
                    image_source = 'Visual7W'
                    image_path = f"{self.image_path}/{img_file}"
                    assert exists(image_path)
                    out_dict = {'unique_id': unique_id,
                        'image_source': image_source,
                        'task_name':  self.task_type,
                        'image_path': image_path,
                        'region': [answer],
                        'question': qa_pair['question'],
                        'options': choices,
                        'meta_data': {'object': bbox[qa_pair['answer']]['obj'], 'question_type': qa_pair['type']}
                        }
                    all_qa_pairs.append(out_dict)
        with open(save_fn, 'w') as fout:
            count = 0
            for ex in all_qa_pairs:
                fout.write(json.dumps(ex)+'\n')
                count += 1
                if count == self.num_test:
                    return
        # print(f"number of skipped instance {skip}")
            
    def create_data(self):

        save_fn = self.out_data_dir / f'test.jsonl'
        
        test_insts = []
        if self.task_type == "grounded_VQA":
            self.create_grounded_VQA_data(save_fn)
            
        else:
            raise NotImplemented

if __name__ == '__main__':
    arg_parser = ArgumentParser(
        'Create dataset for Text VQA')
    arg_parser.add_argument('--out_data_dir', type=Path,
                            default='unseen_data/text_vqa',
                            help='Path to the output data folder')
    arg_parser.add_argument('--task_type', type=str,
                            default='ocr',
                            choices=['grounded_VQA', 'telling_grounded_VQA'],
                            help='Specify the type of task to create dataset')
    arg_parser.add_argument('--num_test', type=int,
                            default=300,
                            help='number of testing instance')
    args = arg_parser.parse_args()

    dataset = InstructData(args)
    dataset.create_data()