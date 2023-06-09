"""answer question base on the content of the image"""
from argparse import ArgumentParser
import json
from pathlib import Path
import random
import os
from os.path import exists
import pdb

class InstructData:

    def __init__(self, args):
        self.train_dir = './raw_datasets/TDIUC'
        self.val_dir = './raw_datasets/TDIUC'
        self.train_img_dir = './raw_datasets/MSCOCO2014/train2014'
        self.val_img_dir = './raw_datasets/MSCOCO2014/val2014'
        self.task_type = ['object_presence', 'sport_recognition', 'color', 'counting', 'sentiment_understanding', 'absurd', 'scene_recognition', 'positional_reasoning', 'activity_recognition', 'attribute', 'object_recognition', 'utility_affordance']
        self.out_data_dir = Path('training_data')
        # self.out_data_dir.mkdir(parents=True, exist_ok=True)
        self.meta_info_fn = 'meta.json'
        self.num_train = args.num_train
        self.num_val = args.num_val


    def merge_annotation(self, questions, answers):
        candidate_dict = {}
        a_dict = {}
        for a in answers:
            assert not a['question_id'] in a_dict
            a_dict[a['question_id']] = a
        question_type_dict = {}
        for q in questions:
            a = a_dict[q['question_id']]
            q['answer'] = a
            question_type = a['question_type']
            assert a['image_id'] == q['image_id']
            if not question_type in question_type_dict:
                question_type_dict[question_type] = []
                candidate_dict[question_type] = set([])
            question_type_dict[question_type].append(q)
            answers = a['answers']
            assert len(answers) == 1
            answer = answers[0]['answer']
            candidate_dict[question_type].add(answer)
        return question_type_dict, candidate_dict
            
    def create_image_understanding_challanges_data(self, questions, answers, save_fn, split, num_inst):
        
        type_annotation_dict, candidate_dict = self.merge_annotation(questions, answers)
        
        """
        {'image_id': 5277, 'question': 'Are there any couches in the photo?', 'question_id': 10026521, 'answer': {'question_type': 'object_presence', 'image_id': 5277, 'ans_source': 'generation', 'answers': [{'answer': 'no', 'answer_confidence': 'yes', 'answer_id': 1}], 'question_id': 10026521}}
        """
        for k, annotations in type_annotation_dict.items():
            out_data_dir = self.out_data_dir / f"TDIUC_{k}"
            out_data_dir.mkdir(parents=True, exist_ok=True)
            save_fn = out_data_dir / f"{split}.jsonl"
            candidates = candidate_dict[k]
            count = 0
            print(k, len(candidates))
            with open(save_fn, 'w') as fout:
                for line in annotations:
                    question = line['question']
                    target_txt = line['answer']['answers'][0]['answer']
                    options = random.sample(list(candidates), 8) if len(candidates) >= 8 else random.sample(list(candidates), len(candidates))
                    if not target_txt in options:
                        options.append(target_txt)
                    random.shuffle(options)
                    if split == 'train':
                        image_path = os.path.join(self.train_img_dir,f"COCO_train2014_{line['image_id']:012d}.jpg")
                    elif split == 'valid':
                        image_path = os.path.join(self.val_img_dir,f"COCO_val2014_{line['image_id']:012d}.jpg")
                    if not exists(image_path):
                        # print(image_path)
                        continue
                    unique_id = f"VQA_{k}_{line['question_id']}"
                    image_source = 'coco_2014'
                    out_dict = {'unique_id': unique_id,
                    'task_name':  f"VQA_{k}",
                    'image_path':image_path,
                    'question': question,
                    'target_txt': target_txt,
                    'options': options
                    }
                    fout.write(json.dumps(out_dict)+'\n')
                    
                    count += 1
                    if count >= num_inst:
                        print(count)
                        break
                
                
        # random.shuffle(inputs)
        # outputs = []
        # count = 0
        # for line in inputs:
        #     text = line['caption']
        #     target_txt = 'yes' if line['label'] == 1 else 'no'
        #     image_path = os.path.join(self.train_image_path,f"{line['image']}")
        #     exists(image_path)
        #     unique_id = f"visual_spatial_reasoning_{split}_{count}"
        #     out_dict = {'unique_id': unique_id,
        #         'task_name':  self.task_type,
        #         'image_path':image_path,
        #         'text': text,
        #         'target_txt': target_txt
        #         }
        #     count += 1
        #     outputs.append(out_dict)
        # # random.shuffle(outputs)
        # with open(save_fn, 'w') as fout:
        #     count = 0
        #     for ex in outputs:
        #         fout.write(json.dumps(ex)+'\n')
        #         count += 1
        #         if count == num_inst:
        #             return        
            
    def create_data(self):
        
        train_q = json.load(open(os.path.join(self.train_dir, 'Questions/OpenEnded_mscoco_train2014_questions.json'),'rb'))
        train_q = train_q['questions']
        train_a = json.load(open(os.path.join(self.train_dir, 'Annotations/mscoco_train2014_annotations.json'),'rb'))
        train_a = train_a['annotations']
        
        val_q = json.load(open(os.path.join(self.train_dir, 'Questions/OpenEnded_mscoco_val2014_questions.json'),'rb'))
        val_q = val_q['questions']
        val_a = json.load(open(os.path.join(self.train_dir, 'Annotations/mscoco_val2014_annotations.json'),'rb'))
        val_a = val_a['annotations']
        
        self.create_image_understanding_challanges_data(train_q, train_a, '', 'train', self.num_train)
        
        self.create_image_understanding_challanges_data(val_q, val_a, '', 'valid', self.num_val)
                


if __name__ == '__main__':
    arg_parser = ArgumentParser(
        'Create dataset for Hateful Memes')
    arg_parser.add_argument('--num_train', type=int,
                            help='number of training instance')
    arg_parser.add_argument('--num_val', type=int,
                            help='number of dev instance')
    args = arg_parser.parse_args()

    dataset = InstructData(args)
    dataset.create_data()