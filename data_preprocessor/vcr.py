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
        self.data_dir = Path('./raw_datasets/VCR/')
        self.out_data_dir = args.out_data_dir
        self.out_data_dir.mkdir(parents=True, exist_ok=True)
        self.task_type = args.task_type
        self.meta_info_fn = 'meta.json'
        self.num_test = args.num_test
        self.image_path = self.data_dir
        self.split = 'val'
        self.annotation_file = self.data_dir / f"{self.split}.jsonl"
    
    def reformat_objects(self, objects):
        obj_dict = {}
        object_lst = []
        for obj in objects:
            if obj not in obj_dict:
                obj_dict[obj] = 1
            else:
                obj_dict[obj] += 1
                
            object_lst.append(f'[{obj} {obj_dict[obj]}]')
        
        return object_lst

    def format_text(self, text_list, object_lst):
        text = ""
        for txt in text_list:
            if type(txt) == str:
                text += txt + ' '
            else: # list
                replaced_txt = [object_lst[t] for t in txt]
                obj_text = " and ".join(replaced_txt)
                text += obj_text + ' '
        return text.strip()
    
    def get_mentioned_obj_ids(self, qa_lst):
        mentioned_obj = set()
        for sentence in qa_lst:
            for word in sentence:
                if type(word) is list:
                    for idx in word:
                        mentioned_obj.add(idx)
        
        return list(mentioned_obj)
                    
        
    
    def create_qa_data(self, save_fn):
        # Q -> A
        with open(self.annotation_file, 'r') as fin:
            
            with open(save_fn, 'w') as fout:
                count = 0
                for line in fin:
                    ex = json.loads(line)
                    image_id = ex['img_id']
                    unique_id = f"vcr_{image_id}"
                    image_path = str(self.data_dir / 'vcr1images' / ex['img_fn'])
                    assert exists(image_path)
                    mentioned_obj_ids = self.get_mentioned_obj_ids(ex['answer_choices'] + [ex['question']])
                    object_lst = self.reformat_objects(ex['objects'])
                    question = self.format_text(ex['question'], object_lst)
                    answers = [self.format_text(a, object_lst) for a in ex['answer_choices']]
                    answer = answers[ex['answer_label']]
                    # rationales = [self.format_text(a, object_lst) for a in ex['rationale_choices']]
                    # rational = answers[ex['rationale_label']]
                    with open(self.data_dir / 'vcr1images' /  ex['metadata_fn'], 'r') as f:
                        boxes = json.load(f)['boxes']
                    object_regions = {}
                    object_regions = {}
                    for idx in mentioned_obj_ids:
                        obj, box = object_lst[idx], boxes[idx]
                        object_regions[obj] = box[:-1]
                    image_source = 'vcr'
                    
                    new_object_region = {}
                    for k, v in object_regions.items():
                        if k in question:
                            question = question.replace(k, f"{k[1:-1]}")
                        new_object_region[k[1:-1]] = v
                        for i, op in enumerate(answers):
                            if k in op:
                                answers[i] = answers[i].replace(k, f"{k[1:-1]}")
                        if k in answer:
                            answer = answer.replace(k, f"{k[1:-1]}")
                    out_dict = {'unique_id': unique_id,
                                'image_source': image_source,
                                'task_name':  self.task_type,
                                'image_path': image_path,
                                'target_txt': answer,
                                'question': question,
                                'options': answers,
                                'meta_data': {'object_regions': new_object_region}
                                }

                    fout.write(json.dumps(out_dict)+'\n')
                    count += 1
                    if count == self.num_test:
                        return

    def create_ar_data(self, save_fn):
        # QA -> R
        with open(self.annotation_file, 'r') as fin:
            
            with open(save_fn, 'w') as fout:
                count = 0
                for line in fin:
                    ex = json.loads(line)
                    image_id = ex['img_id']
                    unique_id = f"vcr_{image_id}"
                    image_path = str(self.data_dir / 'vcr1images' / ex['img_fn'])
                    assert exists(image_path)
                    mentioned_obj_ids = self.get_mentioned_obj_ids(ex['answer_choices'] + ex['rationale_choices'] + [ex['question']])
                    object_lst = self.reformat_objects(ex['objects'])
                    question = self.format_text(ex['question'], object_lst)
                    answers = [self.format_text(a, object_lst) for a in ex['answer_choices']]
                    answer = answers[ex['answer_label']]
                    rationales = [self.format_text(a, object_lst) for a in ex['rationale_choices']]
                    rational = rationales[ex['rationale_label']]
                    with open(self.data_dir / 'vcr1images' /  ex['metadata_fn'], 'r') as f:
                        boxes = json.load(f)['boxes']
                    object_regions = {}
                    for idx in mentioned_obj_ids:
                        obj, box = object_lst[idx], boxes[idx]
                        object_regions[obj] = box[:-1]
                    
                    image_source = 'vcr'
                    out_dict = {'unique_id': unique_id,
                                'image_source': image_source,
                                'task_name':  self.task_type,
                                'image_path': image_path,
                                'target_txt': rational,
                                'question': question,
                                'answer': answer, # new field!
                                'options': rationales,
                                'meta_data': {'object_regions': object_regions}
                                }

                    fout.write(json.dumps(out_dict)+'\n')
                    count += 1
                    if count == self.num_test:
                        return

  
    def create_data(self):

        meta_data = {
            "originial_data_dir": str(self.data_dir / f'vcr1images')
        }
        
        with open(self.out_data_dir / self.meta_info_fn, 'w') as f:
            json.dump(meta_data, f)

        save_fn = self.out_data_dir / f'test.jsonl'
        
        if self.task_type == 'commonsense_VQA':
            self.create_qa_data(save_fn)    
        elif self.task_type == 'visual_answer_justification':
            self.create_ar_data(save_fn)
        elif self.task_type == 'qar':
            raise NotImplemented
        else:
            raise NotImplemented
            
        
        

if __name__ == '__main__':
    arg_parser = ArgumentParser(
        'Create dataset for Visual Commonsense Reasoning')
    arg_parser.add_argument('--out_data_dir', type=Path,
                            default='unseen_data/vcr',
                            help='Path to the output data folder')
    arg_parser.add_argument('--task_type', type=str,
                            default='grounded_VQA',
                            choices=['commonsense_VQA', 'visual_answer_justification', 'qar'],
                            help='Specify the type of task to create dataset: [qa(question answering Q->A), ar(answer justification QA->R), qar(Q->AR)]')
    arg_parser.add_argument('--num_test', type=int,
                            default=300,
                            help='number of testing instance')
    args = arg_parser.parse_args()

    dataset = InstructData(args)
    dataset.create_data()