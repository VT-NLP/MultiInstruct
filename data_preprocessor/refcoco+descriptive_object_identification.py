"""given the description of the object, identify the region Refcoco"""

from argparse import ArgumentParser
import json
from pathlib import Path
import pdb
# import pandas as pd
import os
import random
from os.path import exists
import pickle
import pdb

class InstructData:

    def __init__(self, args):
        self.train_dir = './raw_datasets'
        self.train_datasets=['refcoco/refcoco','refcoco+/refcoco+/','refcocog/refcocog']
        self.train_annotations=['refs(google).p','refs(unc).p','refs(google).p']
        self.train_img_dir = './raw_datasets/MSCOCO2014/train2014'
        self.val_img_dir = './raw_datasets/MSCOCO2014/val2014'
        
        self.task_type = args.task_type
        self.out_data_dir = args.out_data_dir
        # self.out_data_dir.mkdir(parents=True, exist_ok=True)
        self.meta_info_fn = 'meta.json'
        self.num_train = args.num_train
        self.num_val = args.num_val
        # self.datasets = ['refcocog']
        # self.splits = ['refs(google).p']

    def create_descriptive_object_region_generate(self,input_data, image2boxs, out_data_dir, num_inst, data_type='train'):
        save_path = out_data_dir / f'{data_type}.jsonl'
        with open(save_path,'w') as fout:
            count = 0
            # pdb.set_trace()
            for line in input_data:
                if (data_type == 'train' and line['split'] == 'train') or (data_type == 'valid' and line['split'] == 'val') :
                    ann_id = line['ann_id']
                    sent_ids = line['sent_ids']
                    ref_id = line['ref_id']
                    image_id = line['image_id']
                    for sent, sent_id in zip(line['sentences'], sent_ids):
                        sent = sent['raw']
                        unique_id = f"{image_id}_{ann_id}_{ref_id}_{sent_id}"
                        image_source = 'coco2014'
                        image_path = os.path.join(self.train_img_dir,f"COCO_train2014_{line['image_id']:012d}.jpg")
                        assert exists(image_path)
                        region = line['bbox']['bbox']
                        out_dict = {'unique_id': unique_id,
                                    'image_source': image_source,
                                    'task_name':  self.task_type,
                                    'image_path': image_path,
                                    'region': [region],
                                    'text':sent
                        }
                        fout.write(json.dumps(out_dict)+'\n')
                        count+=1
                        if count == num_inst:
                            return
                        
    def create_descriptive_object_region_select(self,input_data, image2boxs, out_data_dir, num_inst, data_type='train'):
        save_path = out_data_dir / f'{data_type}.jsonl'
        with open(save_path,'w') as fout:
            count = 0
            for line in input_data:
                if (data_type == 'train' and line['split'] == 'train') or (data_type == 'valid' and line['split'] == 'val') :
                    ann_id = line['ann_id']
                    sent_ids = line['sent_ids']
                    ref_id = line['ref_id']
                    image_id = line['image_id']
                    all_bboxs = image2boxs[image_id]
                    if not len(all_bboxs) > 1:
                        continue
                    options = [box['bbox'] for box in all_bboxs]
                    for sent, sent_id in zip(line['sentences'], sent_ids):
                        sent = sent['raw']
                        unique_id = f"{image_id}_{ann_id}_{ref_id}_{sent_id}"
                        image_source = 'coco2014'
                        image_path = os.path.join(self.train_img_dir,f"COCO_train2014_{line['image_id']:012d}.jpg")
                        assert exists(image_path)
                        region = line['bbox']['bbox']
                        assert region in options
                        out_dict = {'unique_id': unique_id,
                                    'image_source': image_source,
                                    'task_name':  self.task_type,
                                    'image_path': image_path,
                                    'region': [region],
                                    'options': options,
                                    'text':sent
                        }
                        fout.write(json.dumps(out_dict)+'\n')
                        count+=1
                        if count == num_inst:
                            return
    
    def create_object_description_generate(self,input_data, image2text, out_data_dir, num_inst, data_type='train'):
        save_path = out_data_dir / f'{data_type}.jsonl'
        with open(save_path,'w') as fout:
            count = 0
            for line in input_data:
                if (data_type == 'train' and line['split'] == 'train') or (data_type == 'valid' and line['split'] == 'val') :
                    ann_id = line['ann_id']
                    sent_ids = line['sent_ids']
                    ref_id = line['ref_id']
                    image_id = line['image_id']
                    for sent, sent_id in zip(line['sentences'], sent_ids):
                        sent = sent['raw']
                        unique_id = f"{image_id}_{ann_id}_{ref_id}_{sent_id}"
                        image_source = 'coco2014'
                        image_path = os.path.join(self.train_img_dir,f"COCO_train2014_{line['image_id']:012d}.jpg")
                        assert exists(image_path)
                        region = line['bbox']['bbox']
                        out_dict = {'unique_id': unique_id,
                                    'image_source': image_source,
                                    'task_name':  self.task_type,
                                    'image_path': image_path,
                                    'region': [region],
                                    'target_txt':sent
                        }
                        fout.write(json.dumps(out_dict)+'\n')
                        count+=1
                        if count == num_inst:
                            return
                        
    # def create_object_description_select(self,input_data, image2text, out_data_dir, num_inst, data_type='train'):
    #     save_path = out_data_dir / f'{data_type}.jsonl'
    #     with open(save_path,'w') as fout:
    #         count = 0
    #         for line in input_data:
    #             if (data_type == 'train' and line['split'] == 'train') or (data_type == 'valid' and line['split'] == 'val') :
    #                 ann_id = line['ann_id']
    #                 sent_ids = line['sent_ids']
    #                 ref_id = line['ref_id']
    #                 image_id = line['image_id']
    #                 if len(image2text[image_id]) < 2:
    #                     continue
                    
    #                 options = []
    #                 for img_txt in image2text[image_id]:
    #                     # pdb.set_trace()
    #                     if not img_txt['ann_id'] == ann_id:
    #                         for sent in img_txt['sentences']:
    #                             options.append(sent['raw'])
                                    
    #                 for sent, sent_id in zip(line['sentences'], sent_ids):
    #                     sent = sent['raw']
    #                     unique_id = f"{image_id}_{ann_id}_{ref_id}_{sent_id}"
    #                     image_source = 'coco2014'
    #                     image_path = os.path.join(self.train_img_dir,f"COCO_train2014_{line['image_id']:012d}.jpg")
    #                     assert exists(image_path)
    #                     region = line['bbox']['bbox']
    #                     out_dict = {'unique_id': unique_id,
    #                                 'image_source': image_source,
    #                                 'task_name':  self.task_type,
    #                                 'image_path': image_path,
    #                                 'region': [region],
    #                                 'options': options+[sent],
    #                                 'target_txt':sent
    #                     }
    #                     if len(options) > 5:
    #                         pdb.set_trace()
    #                     fout.write(json.dumps(out_dict)+'\n')
    #                     count+=1
    #                     if count == num_inst:
    #                         return
                        
    

    def create_data(self):

        # meta_data = {
        #     "originial_data_dir": str(self.train_dir)
        # }
        
        # with open(self.out_data_dir / self.meta_info_fn, 'w') as f:
        #     json.dump(meta_data, f)

        
        for dataset, annotation_file in zip(self.train_datasets,self.train_annotations):
            print(dataset)
            bbxs = {}
            image2boxs = {}
            input_data = []
            image2text = {}
            
            raw_instances =  json.load(open(f'{self.train_dir}/{dataset}/instances.json','r'))
            annotations = pickle.load(open(f'{self.train_dir}/{dataset}/{annotation_file}','rb'))
            
            instances = []
            for line in raw_instances['annotations']:
                
                x,y,w,h = line['bbox']
                line['bbox'] = [x,y,x+w,y+h]
                instances.append(line)
            
            for line in instances:
                assert not line['id'] in bbxs
                bbxs[line['id']] = line
                
            for line in instances:
                image_id = line['image_id']
                if not image_id in image2boxs:
                    image2boxs[image_id] = [line]
                else:
                    image2boxs[image_id].append(line)

            for line in annotations:
                ref_id = line['ann_id']
                box = bbxs[ref_id]
                assert box['image_id'] == line['image_id']
                line['bbox'] = box
                input_data.append(line)
                
            for line in annotations:
                # ref_id = line['ann_id']
                # box = bbxs[ref_id]
                # assert box['image_id'] == line['image_id']
                # line['bbox'] = box
                # input_data.append(line)
                if not line['image_id'] in image2text:
                    image2text[line['image_id']] = [line]
                else:
                    image2text[line['image_id']].append(line)
            # pdb.set_trace()
            dataset_name = dataset.split('/')[0]
            out_data_dir = Path(str(self.out_data_dir).replace('refcoco',dataset_name))
            out_data_dir.mkdir(parents=True, exist_ok=True)
            if self.task_type == 'descriptive_object_region_generate':
                self.create_descriptive_object_region_generate(input_data, image2boxs, out_data_dir, self.num_train, data_type='train')
                self.create_descriptive_object_region_generate(input_data, image2boxs, out_data_dir, self.num_val, data_type='valid')
            elif self.task_type == 'descriptive_object_region_select':
                self.create_descriptive_object_region_select(input_data, image2boxs, out_data_dir, self.num_train, data_type='train')
                self.create_descriptive_object_region_select(input_data, image2boxs, out_data_dir, self.num_val, data_type='valid')
            elif self.task_type == 'object_description_generate':
                self.create_object_description_generate(input_data, image2text, out_data_dir, self.num_train, data_type='train')
                self.create_object_description_generate(input_data, image2text, out_data_dir, self.num_val, data_type='valid')
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