"""decide if the question can be answer by the given image"""
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
        self.vocabs = [['yes','no'],['I can answer the question based on the image',"I can not anser the question based on the image"],['possible','not possible'],['the question is relevant to the image','the question is irrelevant to the image']]


    def create_question_image_match_inst(self, save_path, questions, answers, split):
        with open(save_path,'w') as fout:
            count = 0
            for q, a in zip(questions, answers):
                assert a['question_id'] == q['question_id']
                if split == 'valid':  
                    image_path = os.path.join(self.val_img_dir , f"COCO_val2014_{q['image_id']:012d}.jpg")
                else:
                    image_path = os.path.join(self.train_img_dir , f"COCO_train2014_{q['image_id']:012d}.jpg")
                assert exists(image_path)
                vocabs = random.choice(self.vocabs)
                if random.uniform(0,1)> 0.5:
                    out_dict = {'unique_id': 'VQAv2_'+str(q['question_id']),
                                'image_source': 'coco2014',
                                'task_name':  self.task_type,
                                'question': q['question'],
                                'target_txt': vocabs[0],
                                'image_path': image_path,
                                'answer': a['multiple_choice_answer'],
                                'options':vocabs
                    }
                else:
                    # sample a question
                    while True:
                        wrong_q = random.choice(questions)
                        if not wrong_q['image_id'] == q['image_id']:
                            break
                    out_dict = {'unique_id': 'VQAv2_'+str(q['question_id']),
                                'image_source': 'coco2014',
                                'task_name':  self.task_type,
                                'question': wrong_q['question'],
                                'target_txt': vocabs[1],
                                'image_path': image_path,
                                'answer': a['multiple_choice_answer'],
                                'options':vocabs
                    }
                fout.write(json.dumps(out_dict)+'\n')
                count += 1
                if split == 'train':
                    if count == self.num_train:
                        break
                else:
                    if count == self.num_val:
                        break
        
    def create_data(self):

        meta_data = {
            "originial_data_dir": str(self.train_dir)
        }
        with open(self.out_data_dir / self.meta_info_fn, 'w') as f:
            json.dump(meta_data, f)
        if self.task_type == 'question_image_match':
            save_path = self.out_data_dir / 'valid.jsonl'
            questions = json.load(open(os.path.join(self.val_dir,'v2_OpenEnded_mscoco_val2014_questions.json'),'r'))
            questions = questions['questions']
            answers = json.load(open(os.path.join(self.val_dir,'v2_mscoco_val2014_annotations.json'),'r'))
            answers = answers['annotations']
            temp = list(zip(questions, answers))
            temp = random.sample(temp, len(temp))
            questions, answers = zip(*temp)
            questions, answers = list(questions), list(answers)
            self.create_question_image_match_inst(save_path, questions, answers, 'valid')

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
            self.create_question_image_match_inst(save_path, questions, answers, 'train')


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