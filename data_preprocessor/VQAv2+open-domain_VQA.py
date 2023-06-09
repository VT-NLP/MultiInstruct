"""answer question base on the content of the image"""
from argparse import ArgumentParser
import json
from pathlib import Path
import random
import os
from os.path import exists

class InstructData:

    def __init__(self, args):
        self.train_dir = './raw_datasets/VQA_V2'
        self.val_dir = './raw_datasets/VQA_V2'
        self.train_img_dir = './raw_datasets/MSCOCO2014/train2014'
        self.val_img_dir = './raw_datasets/MSCOCO2014/val2014'
        self.task_type = args.task_type
        self.out_data_dir = args.out_data_dir
        self.out_data_dir.mkdir(parents=True, exist_ok=True)
        self.meta_info_fn = 'meta.json'
        self.num_train = args.num_train
        self.num_val = args.num_val

    def create_data(self):

        meta_data = {
            "originial_data_dir": str(self.train_dir)
        }
        with open(self.out_data_dir / self.meta_info_fn, 'w') as f:
            json.dump(meta_data, f)

        
        if self.task_type == 'open-domain_VQA':
            save_path = self.out_data_dir / 'valid.jsonl'

            questions = json.load(open(os.path.join(self.val_dir,'v2_OpenEnded_mscoco_val2014_questions.json'),'r'))
            questions = questions['questions']
            answers = json.load(open(os.path.join(self.val_dir,'v2_mscoco_val2014_annotations.json'),'r'))
            answers = answers['annotations']

            temp = list(zip(questions, answers))
            temp = random.sample(temp, self.num_val)
            questions, answers = zip(*temp)
            questions, answers = list(questions), list(answers)

            with open(save_path,'w') as fout:
                for q, a in zip(questions, answers):
                    assert a['question_id'] == q['question_id']
                    image_path = os.path.join(self.val_img_dir , f"COCO_val2014_{q['image_id']:012d}.jpg")
                    assert exists(image_path)
                    out_dict = {'unique_id': 'VQAv2_'+str(q['question_id']),
                                'image_source': 'coco2014',
                                'task_name':  self.task_type,
                                'question': q['question'],
                                'target_txt': a['multiple_choice_answer'],
                                'image_path': os.path.join(self.val_img_dir , f"COCO_val2014_{q['image_id']:012d}.jpg")
                    }
                    fout.write(json.dumps(out_dict)+'\n')


            save_path = self.out_data_dir / 'train.jsonl'
            questions = json.load(open(os.path.join(self.train_dir,'v2_OpenEnded_mscoco_train2014_questions.json'),'r'))
            questions = questions['questions']
            answers = json.load(open(os.path.join(self.train_dir,'v2_mscoco_train2014_annotations.json'),'r'))
            answers = answers['annotations']

            temp = list(zip(questions, answers))
            if not self.num_train == -1:
                temp = random.sample(temp, self.num_train)
            questions, answers = zip(*temp)
            questions, answers = list(questions), list(answers)

            with open(save_path,'w') as fout:
                for q, a in zip(questions, answers):
                    assert a['question_id'] == q['question_id']
                    image_path = os.path.join(self.train_img_dir , f"COCO_train2014_{q['image_id']:012d}.jpg")
                    assert exists(image_path)

                    out_dict = {'unique_id': 'VQAv2_'+str(q['question_id']),
                                'image_source': 'coco2014',
                                'task_name':  self.task_type,
                                'question': q['question'],
                                'target_txt': a['multiple_choice_answer'],
                                'image_path': os.path.join(self.train_img_dir , f"COCO_train2014_{q['image_id']:012d}.jpg")
                    }
                    fout.write(json.dumps(out_dict)+'\n')
                


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