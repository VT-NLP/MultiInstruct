from argparse import ArgumentParser
import json
from pathlib import Path
import os
import random
from os.path import exists
import pdb

class InstructData:

    def __init__(self, args):
        self.data_dir = Path('./raw_datasets/visdial/')
        self.out_data_dir = args.out_data_dir 
        self.out_data_dir.mkdir(parents=True, exist_ok=True)
        self.meta_info_fn = 'meta.json'
        self.num_test = args.num_test
        self.image_path = self.data_dir / 'VisualDialog_val2018'
        self.task_type = args.task_type
        self.split = 'val'
        self.annotation_file = self.data_dir / 'visdial_1.0_val.json'
    
            
    def create_data(self):

        save_fn = self.out_data_dir / f'test.jsonl'
            
        with open(self.annotation_file, 'r') as f:
            data = json.load(f)['data']
        
        question_lst = data['questions']
        answer_lst = data['answers']
        with open(save_fn, 'w') as fout:
            count = 0
            for ex in data['dialogs']:
                image_source = 'visdial'
                img_id = ex['image_id']
                unique_id = f"visdial_{self.split}_{img_id}"
                image_path = self.image_path / f"VisualDialog_val2018_{img_id:012d}.jpg"
                assert exists(image_path)
                caption = ex['caption']
                dialog_hist = []
                for turn in ex['dialog']:
                    q = question_lst[turn['question']]
                    a = answer_lst[turn['answer']]
                    out_dict = {'unique_id': unique_id,
                                'image_source': image_source,
                                'task_name':  self.task_type,
                                'image_path': str(image_path),
                                'question': q,
                                'target_txt': a,
                                'meta_data': {'caption':caption, 'dialog_hist': dialog_hist}
                    }
                    fout.write(json.dumps(out_dict)+'\n')
                    count+=1
                    dialog_hist.append({'q': q, 'a': a})
                    if count == self.num_test:
                        return

if __name__ == '__main__':
    arg_parser = ArgumentParser(
        'Create dataset for Visual Dialog')
    arg_parser.add_argument('--out_data_dir', type=Path,
                            default='unseen_data/visdial',
                            help='Path to the output data folder')
    arg_parser.add_argument('--task_type', type=str,
                            default='visual_dialog',
                            help='Specify the type of task to create dataset')
    arg_parser.add_argument('--num_test', type=int,
                            help='number of testing instance')
    args = arg_parser.parse_args()

    dataset = InstructData(args)
    dataset.create_data()