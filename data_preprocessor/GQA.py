"""{'semantic': [{'operation': 'select', 'dependencies': [], 'argument': 'scene'}, {'operation': 'query', 'dependencies': [0], 'argument': 'place'}], 'entailed': ['18905189', '18905190'], 'equivalent': ['18905191'], 'question': 'Which place is it?', 'imageId': '2379303', 'isBalanced': True, 'groups': {'global': 'place', 'local': '02q-place'}, 'answer': 'swimming pool', 'semanticStr': 'select: scene->query: place [0]', 'annotations': {'answer': {}, 'question': {}, 'fullAnswer': {}}, 'types': {'detailed': 'place', 'semantic': 'global', 'structural': 'query'}, 'fullAnswer': 'It is a swimming pool.'}"""
"images/2379303.jpg"

from argparse import ArgumentParser
import json
from pathlib import Path
# import pdb
# # import pandas as pd
import os
import random
from os.path import exists

class InstructData:

    def __init__(self, args):
        self.train_dir = './raw_datasets/GQA/train_balanced_questions.json'
        self.val_dir = './raw_datasets/GQA/val_balanced_questions.json'
        self.train_img_dir = './raw_datasets/GQA/images'
        # self.val_img_dir = './raw_datasets/MSCOCO2014/val2014'
        self.task_type = args.task_type
        self.out_data_dir = args.out_data_dir
        self.out_data_dir.mkdir(parents=True, exist_ok=True)
        self.meta_info_fn = 'meta.json'
        self.num_train = args.num_train
        self.num_val = args.num_val

    def create_QA_inst(self,input_data,data_type='train'):
        save_path = self.out_data_dir / f'{data_type}.jsonl'
        with open(save_path,'w') as fout:
            count = 0
            for line in input_data:
                image_id = line['imageId']
                unique_id = f"GQA_{line['unique_id']}_{image_id}"
                question = line['question']
                answer = line['answer']
                image_path = os.path.join(self.train_img_dir,f"{image_id}.jpg")
                assert exists(image_path)
                out_dict = {'unique_id': unique_id,
                            'image_source': "GQA",
                            'task_name':  self.task_type,
                            'image_path': image_path,
                            'question': question,
                            'target_txt':answer
                }
                fout.write(json.dumps(out_dict)+'\n')
                count+=1
                if count == self.num_train and data_type == 'train':
                    print(f'total {data_type} is {count}')
                    return
                elif count == self.num_val and data_type == 'valid':
                    print(f'total {data_type} is {count}')
                    return

    def create_data(self):

        meta_data = {
            "originial_data_dir": str(self.train_dir)
        }
        
        with open(self.out_data_dir / self.meta_info_fn, 'w') as f:
            json.dump(meta_data, f)

        if self.task_type == 'open-domain_VQA':
            valid_input = json.load(open(self.val_dir,'r'))
            valid_data = []
            for k, v in valid_input.items():
                v['unique_id'] = k
                valid_data.append(v)
            random.shuffle(valid_data)
            self.create_QA_inst(valid_data,data_type='valid')
            
            train_input = json.load(open(self.train_dir,'r'))
            train_data = []
            for k, v in train_input.items():
                v['unique_id'] = k
                train_data.append(v)
            random.shuffle(train_data)
            self.create_QA_inst(train_data,data_type='train')
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