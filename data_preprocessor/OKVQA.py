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
        self.data_dir = Path('./raw_datasets/OK-VQA/')
        self.out_data_dir = args.out_data_dir 
        self.out_data_dir.mkdir(parents=True, exist_ok=True)
        self.task_type = 'open-domain_VQA'
        self.meta_info_fn = 'meta.json'
        self.num_train = args.num_train
        self.num_valid = args.num_valid
        self.image_path = Path('./raw_datasets/MSCOCO2014/')
        self.split = ['train','val']
        # self.question_file =  "OpenEnded_mscoco_val2014_questions.json"
        # self.answer_file = "mscoco_val2014_annotations.json"
    
    def gather_answers(self, answers):
        answer_lst = []
        assert len(answers) == 10
        for a in answers:
            answer_lst.append(a['answer'])
        mode_ans = max(set(answer_lst), key=answer_lst.count)
        return answer_lst, mode_ans
        
            
    def create_data(self):

        meta_data = {
            "originial_data_dir": str(self.data_dir / 'val2014')
        }
        
        for split in ['train','val']:
            
            with open(self.out_data_dir / self.meta_info_fn, 'w') as f:
                json.dump(meta_data, f)

            save_fn = self.out_data_dir / f'valid.jsonl' if split == 'val' else self.out_data_dir / f'train.jsonl'
        
            with open(self.data_dir/ split /f"mscoco_{split}2014_annotations.json", 'r') as f:
                # pdb.set_trace()
                answers = json.load(f)['annotations']

            with open(self.data_dir/ split /f"OpenEnded_mscoco_{split}2014_questions.json", 'r') as f:
                questions = json.load(f)['questions']
            
            # pdb.set_trace()
            with open(save_fn, 'w') as fout:
                count = 0
                for q, a in zip(questions, answers):
                    image_id = q['image_id']
                    unique_id = f"ok_vqa_{split}_{image_id}"
                    image_path = str(self.image_path/f'{split}2014' / f"COCO_{split}2014_{image_id:012d}.jpg")
                    if not exists(image_path):
                        pdb.set_trace()
                    image_source = 'mscoco2014'
                    answers, mode_ans = self.gather_answers(a['answers'])
                    out_dict = {'unique_id': unique_id,
                                'image_source': image_source,
                                'task_name':  self.task_type,
                                'image_path': image_path,
                                'target_txt': mode_ans,
                                'question': q['question'],
                                'meta_data': {'answers': answers}
                                }
                    fout.write(json.dumps(out_dict)+'\n')
                    count += 1
                    if split == 'train' and count == self.num_train:
                        break
                    if split == 'val' and count == self.num_valid:
                        break


if __name__ == '__main__':
    arg_parser = ArgumentParser(
        'Create dataset for Outside Knowledge VQA (OK-VQA)')
    arg_parser.add_argument('--out_data_dir', type=Path,
                            default='instruct_data/ok_vqa',
                            help='Path to the output data folder')
    arg_parser.add_argument('--num_train', type=int,
                            default=10000,
                            help='number of testing instance')
    arg_parser.add_argument('--num_valid', type=int,
                            default=1000,
                            help='number of testing instance')
    args = arg_parser.parse_args()

    dataset = InstructData(args)
    dataset.create_data()