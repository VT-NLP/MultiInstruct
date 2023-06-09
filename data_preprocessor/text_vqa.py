"""what is the text in the image"""

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
        self.data_dir = Path('./raw_datasets/TextVQA/train_val/')
        self.out_data_dir = args.out_data_dir
        self.out_data_dir.mkdir(parents=True, exist_ok=True)
        self.task_type = args.task_type
        self.meta_info_fn = 'meta.json'
        self.num_test = args.num_test
        self.image_path = self.data_dir
        self.split = 'val'
        self.annotation_file = self.data_dir / f"TextVQA_0.5.1_{self.split}.json"
        self.ocr_file = self.data_dir / f"TextVQA_Rosetta_OCR_v0.2_{self.split}.json" # words in the image
    
    def gather_answers(self, answers):
        answer_lst = []
        assert len(answers) == 10
        for a in answers:
            answer_lst.append(a['answer'])
        mode_ans = max(set(answer_lst), key=answer_lst.count)
        return answer_lst, mode_ans
    
    def create_vqa_data(self, data, save_fn):
        with open(save_fn, 'w') as fout:
            count = 0
            for ex in data:
                image_id = ex['image_id']
                unique_id = f"text_vqa_{self.split}_{image_id}"
                image_path = str(self.data_dir / 'train_images' / f'{image_id}.jpg')
                assert exists(image_path)
                answers = ex['answers']
                mode_ans = max(set(answers), key=answers.count)
                image_source = 'text_vqa'
                out_dict = {'unique_id': unique_id,
                            'image_source': image_source,
                            'task_name':  self.task_type,
                            'image_path': image_path,
                            'target_txt': mode_ans,
                            'question': ex['question'],
                            'meta_data': {'answers': answers, 'image_classes': ex['image_classes']}
                            }

                fout.write(json.dumps(out_dict)+'\n')
                count += 1
                if count == self.num_test:
                    return

    
    def create_ocr_data(self, data, save_fn):
        with open(self.ocr_file, 'r') as f:
            ocr_data = json.load(f)['data']
            
        ocr_dict = {}
        for ex in ocr_data:
            ocr_dict[ex['image_id']] = {
                'ocr_info': ex['ocr_info']
            }
        with open(save_fn, 'w') as fout:
            count = 0
            for ex in data:
                image_id = ex['image_id']
                unique_id = f"text_vqa_{self.split}_{image_id}"
                image_path = str(self.data_dir / 'train_images' / f'{image_id}.jpg')
                assert exists(image_path)
                image_source = 'text_vqa'
                ocr_tokens = ocr_dict[image_id]['ocr_info']
                width, height = ex['image_width'], ex['image_height']
                for ocr_token in ocr_tokens:
                    bounding_box = ocr_token['bounding_box']
                    region = [bounding_box['top_left_x'] * width, bounding_box['top_left_y'] * height,(bounding_box['top_left_x']+bounding_box['width']) * width,(bounding_box['top_left_y']+bounding_box['height']) * height]
                    if (region[3] - region[1]) * (region[2] - region[0]) <= 6000 * len(ocr_token['word']):
                        continue
                    out_dict = {'unique_id': unique_id,
                                'image_source': image_source,
                                'task_name':  self.task_type,
                                'image_path': image_path,
                                'target_txt': ocr_token['word'],
                                'region': [region],
                                'meta_data': {'bounding_box': bounding_box}
                                }
                    fout.write(json.dumps(out_dict)+'\n')
                    count += 1
                    if count == self.num_test:
                        return
            
    def create_data(self):

        meta_data = {
            "originial_data_dir": str(self.data_dir / f'{self.split}_images')
        }
        
        with open(self.out_data_dir / self.meta_info_fn, 'w') as f:
            json.dump(meta_data, f)

        save_fn = self.out_data_dir / f'test.jsonl'
        
        with open(self.annotation_file, 'r') as f:
            data = json.load(f)['data']
        
        if self.task_type == 'ocr':
            self.create_ocr_data(data, save_fn)
        elif self.task_type == 'text_vqa':
            self.create_vqa_data(data, save_fn)    
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
                            choices=['text_vqa', 'ocr'],
                            help='Specify the type of task to create dataset')
    arg_parser.add_argument('--num_test', type=int,
                            default=300,
                            help='number of testing instance')
    args = arg_parser.parse_args()

    dataset = InstructData(args)
    dataset.create_data()