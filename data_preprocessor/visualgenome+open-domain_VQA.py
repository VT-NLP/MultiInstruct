"""grounded caption task. given region and generate caption for the region """

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
        self.train_dir = './raw_datasets/visual_genome/question_answers.json'
        # self.val_dir = './raw_datasets/MSCOCO2014/annotations/instances_val2014.json'
        self.train_img_dir = './raw_datasets/visual_genome/VG_100K'
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
                for inst in line['qas']:
                    question = inst['question']
                    answer = inst['answer'][:-1]
                    unique_id = f"visualgenome_VG_{str(inst['image_id'])}_{str(inst['qa_id'])}"
                    image_source = 'visualgenome'
                    image_path = os.path.join(self.train_img_dir,f"{str(inst['image_id'])}.jpg")
                    try:
                        assert exists(image_path)
                    except:
                        image_path = os.path.join(self.train_img_dir+'_2',f"{str(inst['image_id'])}.jpg")
                        assert exists(image_path)
                    out_dict = {'unique_id': unique_id,
                                'image_source': image_source,
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
            region_descriptions = json.load(open(self.train_dir,'r'))
            random.shuffle(region_descriptions)
            train_data = region_descriptions[:self.num_train]
            valid_data = region_descriptions[self.num_train:self.num_train+self.num_val]
            self.create_QA_inst(train_data,data_type='train')
            self.create_QA_inst(valid_data,data_type='valid')
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