"""decide th erelationship of two object in the image. the object is specified by the region"""
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
        self.data_dir = '/projects/nlp_lab/zhiyang/projects/datasets/VG_VQA'
        # self.val_dir = './raw_datasets/MSCOCO2014/annotations/instances_val2014.json'
        self.img_dir = './raw_datasets/visual_genome/VG_100K'
        # self.val_img_dir = './raw_datasets/MSCOCO2014/val2014'
        self.task_type = args.task_type
        self.out_data_dir = args.out_data_dir
        self.out_data_dir.mkdir(parents=True, exist_ok=True)
        self.meta_info_fn = 'meta.json'
        self.num_train = args.num_train
        self.num_val = args.num_val

    def create_obj_rel_inst(self,relationships,split,num_inst):
        save_path = self.out_data_dir / f'{split}.jsonl'
        with open(save_path,'w') as fout:
            count = 0
            for line in relationships:
                rels = line['relationships']
                for rel in rels:
                    predicate = rel['predicate'].lower()
                    subject = rel['subject']['names'] if 'names' in rel['subject'] else rel['subject']['name']# list
                    if isinstance(subject, list):
                        assert len(subject) == 1
                        subject = subject[0]
                    object_ = rel['object']['names'] if 'names' in rel['object'] else rel['object']['name']
                    if isinstance(object_, list):
                        assert len(object_) == 1
                        object_ = object_[0]
                    sub_region = [[rel['subject']['x'], rel['subject']['y'], rel['subject']['x']+ rel['subject']['w'], rel['subject']['y']+rel['subject']['h']]] 
                    obj_region = [[rel['object']['x'], rel['object']['y'], rel['object']['x']+ rel['object']['w'], rel['object']['y']+rel['object']['h']]] 

                    unique_id = f"visualgenome_object_relationship_{str(line['image_id'])}_{str(rel['relationship_id'])}"
                    image_source = 'visualgenome'
                    image_path = os.path.join(self.img_dir,f"{str(line['image_id'])}.jpg")
                    try:
                        assert exists(image_path)
                    except:
                        image_path = os.path.join(self.img_dir+'_2',f"{str(line['image_id'])}.jpg")
                        assert exists(image_path)
                    out_dict = {'unique_id': unique_id,
                                'image_source': image_source,
                                'task_name':  self.task_type,
                                'image_path': image_path,
                                'region': sub_region+obj_region,
                                'target_txt': predicate,
                                'meta_data': {'subject':subject, 'object':object_}
                    }
                    fout.write(json.dumps(out_dict)+'\n')
                    count+=1
                    if count == num_inst:
                        return
                
    def create_object_identification_inst(self,relationships,split,num_inst):
        save_path = self.out_data_dir / f'{split}.jsonl'
        with open(save_path,'w') as fout:
            count = 0
            for line in relationships:
                rels = line['relationships']
                for rel in rels:
                    predicate = rel['predicate'].lower()
                    subject = rel['subject']['names'] if 'names' in rel['subject'] else rel['subject']['name']# list
                    if isinstance(subject, list):
                        assert len(subject) == 1
                        subject = subject[0]
                    object_ = rel['object']['names'] if 'names' in rel['object'] else rel['object']['name']
                    if isinstance(object_, list):
                        assert len(object_) == 1
                        object_ = object_[0]
                    sub_region = [[rel['subject']['x'], rel['subject']['y'], rel['subject']['x']+ rel['subject']['w'], rel['subject']['y']+rel['subject']['h']]] 
                    obj_region = [[rel['object']['x'], rel['object']['y'], rel['object']['x']+ rel['object']['w'], rel['object']['y']+rel['object']['h']]] 

                    unique_id = f"visualgenome_object_relationship_{str(line['image_id'])}_{str(rel['relationship_id'])}"
                    image_source = 'visualgenome'
                    image_path = os.path.join(self.img_dir,f"{str(line['image_id'])}.jpg")
                    try:
                        assert exists(image_path)
                    except:
                        image_path = os.path.join(self.img_dir+'_2',f"{str(line['image_id'])}.jpg")
                        assert exists(image_path)
                    out_dict = {'unique_id': unique_id,
                                'image_source': image_source,
                                'task_name':  self.task_type,
                                'image_path': image_path,
                                'region': sub_region+obj_region,
                                'target_txt': object_,
                                'meta_data': {'subject':subject, 'object':object_, 'relation':predicate}
                    }
                    fout.write(json.dumps(out_dict)+'\n')
                    count+=1
                    if count == num_inst:
                        return
                
    def create_subject_identification_inst(self,relationships,split,num_inst):
        save_path = self.out_data_dir / f'{split}.jsonl'
        with open(save_path,'w') as fout:
            count = 0
            for line in relationships:
                rels = line['relationships']
                for rel in rels:
                    predicate = rel['predicate'].lower()
                    subject = rel['subject']['names'] if 'names' in rel['subject'] else rel['subject']['name']# list
                    if isinstance(subject, list):
                        assert len(subject) == 1
                        subject = subject[0]
                    object_ = rel['object']['names'] if 'names' in rel['object'] else rel['object']['name']
                    if isinstance(object_, list):
                        assert len(object_) == 1
                        object_ = object_[0]
                    sub_region = [[rel['subject']['x'], rel['subject']['y'], rel['subject']['x']+ rel['subject']['w'], rel['subject']['y']+rel['subject']['h']]] 
                    obj_region = [[rel['object']['x'], rel['object']['y'], rel['object']['x']+ rel['object']['w'], rel['object']['y']+rel['object']['h']]] 

                    unique_id = f"visualgenome_object_relationship_{str(line['image_id'])}_{str(rel['relationship_id'])}"
                    image_source = 'visualgenome'
                    image_path = os.path.join(self.img_dir,f"{str(line['image_id'])}.jpg")
                    try:
                        assert exists(image_path)
                    except:
                        image_path = os.path.join(self.img_dir+'_2',f"{str(line['image_id'])}.jpg")
                        assert exists(image_path)
                    out_dict = {'unique_id': unique_id,
                                'image_source': image_source,
                                'task_name':  self.task_type,
                                'image_path': image_path,
                                'region': sub_region+obj_region,
                                'target_txt': subject,
                                'meta_data': {'subject':subject, 'object':object_, 'relation':predicate}
                    }
                    fout.write(json.dumps(out_dict)+'\n')
                    count+=1
                    if count == num_inst:
                        return
                
    def create_visual_object_region_inst(self,relationships,split,num_inst):
        save_path = self.out_data_dir / f'{split}.jsonl'
        with open(save_path,'w') as fout:
            count = 0
            for line in relationships:
                rels = line['relationships']
                for rel in rels:
                    predicate = rel['predicate'].lower()
                    subject = rel['subject']['names'] if 'names' in rel['subject'] else rel['subject']['name']# list
                    if isinstance(subject, list):
                        assert len(subject) == 1
                        subject = subject[0]
                    object_ = rel['object']['names'] if 'names' in rel['object'] else rel['object']['name']
                    if isinstance(object_, list):
                        assert len(object_) == 1
                        object_ = object_[0]
                    sub_region = [[rel['subject']['x'], rel['subject']['y'], rel['subject']['x']+ rel['subject']['w'], rel['subject']['y']+rel['subject']['h']]] 
                    obj_region = [[rel['object']['x'], rel['object']['y'], rel['object']['x']+ rel['object']['w'], rel['object']['y']+rel['object']['h']]] 

                    unique_id = f"visualgenome_object_relationship_{str(line['image_id'])}_{str(rel['relationship_id'])}"
                    image_source = 'visualgenome'
                    image_path = os.path.join(self.img_dir,f"{str(line['image_id'])}.jpg")
                    try:
                        assert exists(image_path)
                    except:
                        image_path = os.path.join(self.img_dir+'_2',f"{str(line['image_id'])}.jpg")
                        assert exists(image_path)
                    out_dict = {'unique_id': unique_id,
                                'image_source': image_source,
                                'task_name':  self.task_type,
                                'image_path': image_path,
                                'region': obj_region,
                                'meta_data': {'subject':subject, 'object':object_, 'relation':predicate, 'object_regions':{'subject':sub_region,'object':obj_region}}
                    }
                    fout.write(json.dumps(out_dict)+'\n')
                    count+=1
                    if count == num_inst:
                        return
    
    def create_visual_subject_region_inst(self,relationships,split,num_inst):
        save_path = self.out_data_dir / f'{split}.jsonl'
        with open(save_path,'w') as fout:
            count = 0
            for line in relationships:
                rels = line['relationships']
                for rel in rels:
                    predicate = rel['predicate'].lower()
                    subject = rel['subject']['names'] if 'names' in rel['subject'] else rel['subject']['name']# list
                    if isinstance(subject, list):
                        assert len(subject) == 1
                        subject = subject[0]
                    object_ = rel['object']['names'] if 'names' in rel['object'] else rel['object']['name']
                    if isinstance(object_, list):
                        assert len(object_) == 1
                        object_ = object_[0]
                    sub_region = [[rel['subject']['x'], rel['subject']['y'], rel['subject']['x']+ rel['subject']['w'], rel['subject']['y']+rel['subject']['h']]] 
                    obj_region = [[rel['object']['x'], rel['object']['y'], rel['object']['x']+ rel['object']['w'], rel['object']['y']+rel['object']['h']]] 

                    unique_id = f"visualgenome_object_relationship_{str(line['image_id'])}_{str(rel['relationship_id'])}"
                    image_source = 'visualgenome'
                    image_path = os.path.join(self.img_dir,f"{str(line['image_id'])}.jpg")
                    try:
                        assert exists(image_path)
                    except:
                        image_path = os.path.join(self.img_dir+'_2',f"{str(line['image_id'])}.jpg")
                        assert exists(image_path)
                    out_dict = {'unique_id': unique_id,
                                'image_source': image_source,
                                'task_name':  self.task_type,
                                'image_path': image_path,
                                'region': sub_region,
                                'meta_data': {'subject':subject, 'object':object_, 'relation':predicate, 'object_regions':{'subject':sub_region,'object':obj_region}}
                    }
                    fout.write(json.dumps(out_dict)+'\n')
                    count+=1
                    if count == num_inst:
                        return
                

    def create_data(self):

        meta_data = {
            "originial_data_dir": str(self.data_dir)
        }
        
        with open(self.out_data_dir / self.meta_info_fn, 'w') as f:
            json.dump(meta_data, f)

        relationships = json.load(open(os.path.join(self.data_dir,'relationships.json'),'r'))
        relationships = random.sample(relationships, len(relationships))
        if self.task_type == 'object_relationship':
            self.create_obj_rel_inst(relationships,'train',self.num_train)
            self.create_obj_rel_inst(relationships[self.num_train:],'valid',self.num_val)
        elif self.task_type == 'visual_object_identification':
            self.create_object_identification_inst(relationships,'train',self.num_train)
            self.create_object_identification_inst(relationships[self.num_train:],'valid',self.num_val)
            
        elif self.task_type == 'visual_subject_identification':
            self.create_subject_identification_inst(relationships,'train',self.num_train)
            self.create_subject_identification_inst(relationships[self.num_train:],'valid',self.num_val)
            
        elif self.task_type == 'visual_object_region':
            self.create_visual_object_region_inst(relationships,'train',self.num_train)
            self.create_visual_object_region_inst(relationships[self.num_train:],'valid',self.num_val)
            
        elif self.task_type == 'visual_subject_region':
            self.create_visual_subject_region_inst(relationships,'train',self.num_train)
            self.create_visual_subject_region_inst(relationships[self.num_train:],'valid',self.num_val)
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
                            help='number of testing instance')
    arg_parser.add_argument('--num_val', type=int,
                            help='number of testing instance')
    args = arg_parser.parse_args()

    dataset = InstructData(args)
    dataset.create_data()