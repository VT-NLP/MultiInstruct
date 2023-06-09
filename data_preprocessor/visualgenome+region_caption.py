"""grounded caption task. given region and generate caption for the region """

from argparse import ArgumentParser
import json
from pathlib import Path
import pdb
# import pandas as pd
import os
import random
from os.path import exists

class InstructData:

    def __init__(self, args):
        self.train_dir = './raw_datasets/visual_genome/region_descriptions.json'
        self.train_img_dir = './raw_datasets/visual_genome/VG_100K'
        self.task_type = args.task_type
        self.out_data_dir = args.out_data_dir
        self.out_data_dir.mkdir(parents=True, exist_ok=True)
        self.meta_info_fn = 'meta.json'
        self.num_train = args.num_train
        self.num_val = args.num_val
        self.data_splits = json.load(open('./raw_datasets/visual_genome/VG_splits.json','r'))["VG"]
        self.vocabs = [['yes','no'],['True',"False"],['yes, the text matches the content of the region',"no, the text doesn't matches the content of the region"],["the text describes the content of the region",'the text does not describe the given part of the image']]

    def compute_overlap(self, region1, region2):
        XA1, YA1, XA2, YA2 = region1
        XB1, YB1, XB2, YB2 = region2
        overlap = max(0, min(XA2, XB2) - max(XA1, XB1)) * max(0, min(YA2, YB2) - max(YA1, YB1))
        return overlap
    
    def compute_area(self, region):
        area = (region[2] - region[0]) * (region[3] - region[1])
        return area
        
        
    def create_VG_inst(self,input_data,data_type='train'):
        save_path = self.out_data_dir / f'{data_type}.jsonl'
        with open(save_path,'w') as fout:
            count = 0
            for line in input_data:
                for inst in line['regions']:
                    region = [inst['x'],inst['y'],inst['x']+inst['width'],inst['y']+inst['height']]
                    description = inst['phrase']
                    unique_id = f"visualgenome_VG_{str(inst['image_id'])}_{str(inst['region_id'])}"
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
                                'region': [region],
                                'text':description
                    }
                    fout.write(json.dumps(out_dict)+'\n')
                    count+=1
                    if count == self.num_train and data_type == 'train':
                        return
                    elif count == self.num_val and data_type == 'valid':
                        return
                    
    def create_VG_selection_inst(self,input_data,data_type='train'):
        save_path = self.out_data_dir / f'{data_type}.jsonl'
        with open(save_path,'w') as fout:
            count = 0
            for line in input_data:
                if len(line['regions']) < 3:
                    continue
                else:
                    options = [ [temp['x'],temp['y'],temp['x']+temp['width'],temp['y']+temp['height']] for temp in line['regions']]
                options = random.sample(options, 5) if len(options) > 5 else options
                for inst in line['regions']:
                    region = [inst['x'],inst['y'],inst['x']+inst['width'],inst['y']+inst['height']]
                    description = inst['phrase']
                    unique_id = f"visualgenome_VG_{str(inst['image_id'])}_{str(inst['region_id'])}"
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
                                'region': [region],
                                'options': options + [region] if not region in options else options,
                                'text':description
                    }
                    fout.write(json.dumps(out_dict)+'\n')
                    count+=1
                    if count == self.num_train and data_type == 'train':
                        return
                    elif count == self.num_val and data_type == 'valid':
                        return

    def create_GC_inst(self,input_data,data_type='train'):
        save_path = self.out_data_dir / f'{data_type}.jsonl'
        with open(save_path,'w') as fout:
            count = 0
            for line in input_data:
                for inst in line['regions']:
                    region = [inst['x'],inst['y'],inst['x']+inst['width'],inst['y']+inst['height']]
                    description = inst['phrase']
                    unique_id = f"visualgenome_VG_{str(inst['image_id'])}_{str(inst['region_id'])}"
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
                                'region': [region],
                                'target_txt':description
                    }
                    fout.write(json.dumps(out_dict)+'\n')
                    count+=1
                    if count == self.num_train and data_type == 'train':
                        return
                    elif count == self.num_val and data_type == 'valid':
                        return
    
    def create_GC_selection_inst(self,input_data,data_type='train'):
        save_path = self.out_data_dir / f'{data_type}.jsonl'
        with open(save_path,'w') as fout:
            count = 0
            for line in input_data:
                if len(line['regions']) < 3:
                    continue
                else:
                    options = [ temp['phrase'] for temp in line['regions']]
                options = random.sample(options, 5) if len(options) > 5 else options
                for inst in line['regions']:
                    region = [inst['x'],inst['y'],inst['x']+inst['width'],inst['y']+inst['height']]
                    description = inst['phrase']
                    unique_id = f"visualgenome_VG_{str(inst['image_id'])}_{str(inst['region_id'])}"
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
                                'region': [region],
                                'options': options+[description] if not description in options else options,
                                'target_txt':description
                    }
                    fout.write(json.dumps(out_dict)+'\n')
                    count+=1
                    if count == self.num_train and data_type == 'train':
                        return
                    elif count == self.num_val and data_type == 'valid':
                        return

    def create_region_caption_match_inst(self,input_data,data_type='train'):
        save_path = self.out_data_dir / f'{data_type}.jsonl'
        with open(save_path,'w') as fout:
            count = 0
            for line in input_data:
                if len(line['regions']) < 2:
                    continue
                for inst in line['regions']:
                    region = [inst['x'],inst['y'],inst['x']+inst['width'],inst['y']+inst['height']]
                    
                    vocabs = random.choice(self.vocabs)
                    if random.uniform(0,1) > 0.5:
                        description = inst['phrase']
                        target_txt = vocabs[0]
                    else:
                        num_iter = 0
                        while num_iter < 4:
                            num_iter+=1
                            inst2 = random.choice(line['regions'])
                            if not inst['x']==inst2['x'] or not inst['y']==inst2['y'] or not inst['width']==inst2['width'] or not inst['height']==inst2['height']:
                                break
                        description = inst2['phrase']
                        target_txt = vocabs[1]
                    unique_id = f"visualgenome_VG_{str(inst['image_id'])}_{str(inst['region_id'])}"
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
                                'region': [region],
                                'text':description,
                                'target_txt': target_txt,
                                'meta_data': {'original_caption':inst['phrase']},
                                'options':vocabs
                    }
                    fout.write(json.dumps(out_dict)+'\n')
                    count+=1
                    if count == self.num_train and data_type == 'train':
                        return
                    elif count == self.num_val and data_type == 'valid':
                        return
    
    def create_select_overlap_most_region_inst(self,input_data,data_type='train'):
        save_path = self.out_data_dir / f'{data_type}.jsonl'
        with open(save_path,'w') as fout:
            count = 0
            for line in input_data:
                regions = []
                image_id = None
                for idx, inst in enumerate(line['regions']):
                    region = [inst['x'],inst['y'],inst['x']+inst['width'],inst['y']+inst['height']]
                    description = inst['phrase']
                    unique_id = f"visualgenome_VG_{str(inst['image_id'])}_{str(inst['region_id'])}"
                    image_source = 'visualgenome'
                    image_path = os.path.join(self.train_img_dir,f"{str(inst['image_id'])}.jpg")
                    if idx == 0:
                        image_id = inst['image_id']
                    else:
                        assert inst['image_id'] == image_id
                    exists(image_path)
                    regions.append(region)
                
                if not len(regions) > 3:
                    continue
                
                given_region = random.choice(regions)
                regions.remove(given_region)
                
                max_overlap = 0
                max_idx = None
                temp_regions = []
                for op_idx, op in enumerate(regions):
                    overlap = self.compute_overlap(given_region, op)
                    if (not max_overlap == 0) and  max_overlap == overlap: # make the answer unique
                        continue
                    temp_regions.append(op)
                    if overlap > max_overlap:
                        max_overlap = overlap
                        max_idx = op_idx
                        target_region = op
                if max_overlap == 0:
                    continue
                
                # target_region = regions[max_idx]
                regions.remove(target_region)
                options = random.sample(regions, min(len(regions),3))
                options.append(target_region)
                random.shuffle(options)
                
                
                out_dict = {'unique_id': unique_id,
                            'image_source': image_source,
                            'task_name':  self.task_type,
                            'image_path': image_path,
                            'region': [target_region],
                            'options':options,
                            'meta_data': {'object_regions':{'given_region':[given_region]}}
                }
                fout.write(json.dumps(out_dict)+'\n')
                count+=1
                if count == self.num_train and data_type == 'train':
                    return
                elif count == self.num_val and data_type == 'valid':
                    return
    
    def create_select_overlap_least_region_inst(self,input_data,data_type='train'):
        save_path = self.out_data_dir / f'{data_type}.jsonl'
        with open(save_path,'w') as fout:
            count = 0
            for line in input_data:
                regions = []
                image_id = None
                for idx, inst in enumerate(line['regions']):
                    region = [inst['x'],inst['y'],inst['x']+inst['width'],inst['y']+inst['height']]
                    description = inst['phrase']
                    unique_id = f"visualgenome_VG_{str(inst['image_id'])}_{str(inst['region_id'])}"
                    image_source = 'visualgenome'
                    image_path = os.path.join(self.train_img_dir,f"{str(inst['image_id'])}.jpg")
                    if idx == 0:
                        image_id = inst['image_id']
                    else:
                        assert inst['image_id'] == image_id
                    exists(image_path)
                    regions.append(region)
                
                if not len(regions) > 3:
                    continue
                
                given_region = random.choice(regions)
                regions.remove(given_region)
                overlaped_regions = []
                min_overlap = self.compute_area(given_region)
                min_idx = None
                for op_idx, op in enumerate(regions):
                    overlap = self.compute_overlap(given_region, op)
                    if overlap <= 0 or overlap == min_overlap:
                        continue
                    overlaped_regions.append(op)
                    if overlap < min_overlap:
                        min_overlap = overlap
                        min_idx = op_idx
                        target_region = op
                if min_overlap == self.compute_area(given_region): # no smaller overlap
                    continue
                overlaped_regions.remove(target_region)
                options = random.sample(overlaped_regions, min(len(overlaped_regions),3))
                options.append(target_region)
                random.shuffle(options)
                
                
                out_dict = {'unique_id': unique_id,
                            'image_source': image_source,
                            'task_name':  self.task_type,
                            'image_path': image_path,
                            'region': [target_region],
                            'options':options,
                            'meta_data': {'object_regions':{'given_region':[given_region]}}
                }
                fout.write(json.dumps(out_dict)+'\n')
                count+=1
                if count == self.num_train and data_type == 'train':
                    return
                elif count == self.num_val and data_type == 'valid':
                    return
                
    def create_region_area_inst(self,input_data,data_type='train'):
        save_path = self.out_data_dir / f'{data_type}.jsonl'
        with open(save_path,'w') as fout:
            count = 0
            for line in input_data:
                regions = []
                image_id = None
                for idx, inst in enumerate(line['regions']):
                    region = [inst['x'],inst['y'],inst['x']+inst['width'],inst['y']+inst['height']]
                    description = inst['phrase']
                    unique_id = f"visualgenome_VG_{str(inst['image_id'])}_{str(inst['region_id'])}"
                    image_source = 'visualgenome'
                    image_path = os.path.join(self.train_img_dir,f"{str(inst['image_id'])}.jpg")
                    if idx == 0:
                        image_id = inst['image_id']
                    else:
                        assert inst['image_id'] == image_id
                    exists(image_path)
                    regions.append(region)
                
                given_region = random.choice(regions)
                area = self.compute_area(given_region)
            
                out_dict = {'unique_id': unique_id,
                            'image_source': image_source,
                            'task_name':  self.task_type,
                            'image_path': image_path,
                            'region': [given_region],
                            'target_txt': str(area)
                }
                fout.write(json.dumps(out_dict)+'\n')
                count+=1
                if count == self.num_train and data_type == 'train':
                    return
                elif count == self.num_val and data_type == 'valid':
                    return
                    
    def create_select_overlaped_region_inst(self,input_data,data_type='train'):
        save_path = self.out_data_dir / f'{data_type}.jsonl'
        with open(save_path,'w') as fout:
            count = 0
            for line in input_data:
                regions = []
                image_id = None
                for idx, inst in enumerate(line['regions']):
                    region = [inst['x'],inst['y'],inst['x']+inst['width'],inst['y']+inst['height']]
                    description = inst['phrase']
                    unique_id = f"visualgenome_VG_{str(inst['image_id'])}_{str(inst['region_id'])}"
                    image_source = 'visualgenome'
                    image_path = os.path.join(self.train_img_dir,f"{str(inst['image_id'])}.jpg")
                    if idx == 0:
                        image_id = inst['image_id']
                    else:
                        assert inst['image_id'] == image_id
                    exists(image_path)
                    regions.append(region)
                
                if not len(regions) > 3:
                    continue
                
                given_region = random.choice(regions)
                regions.remove(given_region)
                overlaped_regions = []
                non_overlaped_regions = []
                for op_idx, op in enumerate(regions):
                    overlap = self.compute_overlap(given_region, op)
                    if overlap > 0:
                        overlaped_regions.append(op)
                    else:
                        non_overlaped_regions.append(op)
                if not len(overlaped_regions) > 0 :
                    continue
                if not len(non_overlaped_regions) > 0 :
                    continue
                non_overlaped_regions = random.sample(non_overlaped_regions, min(len(non_overlaped_regions),3))
                overlaped_region = random.choice(overlaped_regions)
                options = [overlaped_region] + non_overlaped_regions
                if len(options) < 2:
                    continue
                random.shuffle(options)
                
                out_dict = {'unique_id': unique_id,
                            'image_source': image_source,
                            'task_name':  self.task_type,
                            'image_path': image_path,
                            'region': [overlaped_region],
                            'options':options,
                            'meta_data': {'object_regions':{'given_region':[given_region]}}
                }
                fout.write(json.dumps(out_dict)+'\n')
                count+=1
                
                if count == self.num_train and data_type == 'train':
                    return
                elif count == self.num_val and data_type == 'valid':
                    return
                
    def create_select_nonoverlaped_region_inst(self,input_data,data_type='train'):
        save_path = self.out_data_dir / f'{data_type}.jsonl'
        with open(save_path,'w') as fout:
            count = 0
            for line in input_data:
                regions = []
                image_id = None
                for idx, inst in enumerate(line['regions']):
                    region = [inst['x'],inst['y'],inst['x']+inst['width'],inst['y']+inst['height']]
                    description = inst['phrase']
                    unique_id = f"visualgenome_VG_{str(inst['image_id'])}_{str(inst['region_id'])}"
                    image_source = 'visualgenome'
                    image_path = os.path.join(self.train_img_dir,f"{str(inst['image_id'])}.jpg")
                    if idx == 0:
                        image_id = inst['image_id']
                    else:
                        assert inst['image_id'] == image_id
                    exists(image_path)
                    regions.append(region)
                
                if not len(regions) > 3:
                    continue
                
                given_region = random.choice(regions)
                regions.remove(given_region)
                overlaped_regions = []
                non_overlaped_regions = []
                for op_idx, op in enumerate(regions):
                    overlap = self.compute_overlap(given_region, op)
                    if overlap > 0:
                        overlaped_regions.append(op)
                    else:
                        non_overlaped_regions.append(op)
                if not len(overlaped_regions) > 0 :
                    continue
                if not len(non_overlaped_regions) > 0 :
                    continue
                
                overlaped_regions = random.sample(overlaped_regions, min(len(overlaped_regions),3))
                non_overlaped_region = random.choice(non_overlaped_regions)
                options = [non_overlaped_region] + overlaped_regions
                if len(options) < 2:
                    continue
                random.shuffle(options)
                
                out_dict = {'unique_id': unique_id,
                            'image_source': image_source,
                            'task_name':  self.task_type,
                            'image_path': image_path,
                            'region': [non_overlaped_region],
                            'options':options,
                            'meta_data': {'object_regions':{'given_region':[given_region]}}
                }
                fout.write(json.dumps(out_dict)+'\n')
                count+=1
                
                if count == self.num_train and data_type == 'train':
                    return
                elif count == self.num_val and data_type == 'valid':
                    return
    
    def create_if_region_overlap_inst(self,input_data,data_type='train'):
        save_path = self.out_data_dir / f'{data_type}.jsonl'
        with open(save_path,'w') as fout:
            count = 0
            for line in input_data:
                regions = []
                image_id = None
                for idx, inst in enumerate(line['regions']):
                    region = [inst['x'],inst['y'],inst['x']+inst['width'],inst['y']+inst['height']]
                    description = inst['phrase']
                    unique_id = f"visualgenome_VG_{str(inst['image_id'])}_{str(inst['region_id'])}"
                    image_source = 'visualgenome'
                    image_path = os.path.join(self.train_img_dir,f"{str(inst['image_id'])}.jpg")
                    if idx == 0:
                        image_id = inst['image_id']
                    else:
                        assert inst['image_id'] == image_id
                    exists(image_path)
                    regions.append(region)
                
                if not len(regions) > 3:
                    continue
                
                given_region = random.choice(regions)
                regions.remove(given_region)
                overlaped_regions = []
                non_overlaped_regions = []
                for op_idx, op in enumerate(regions):
                    overlap = self.compute_overlap(given_region, op)
                    if overlap > 0:
                        overlaped_regions.append(op)
                    else:
                        non_overlaped_regions.append(op)
                if not len(overlaped_regions) > 0 :
                    continue
                if not len(non_overlaped_regions) > 0 :
                    continue
                
                if random.uniform(0, 1) < 0.5:
                    region = random.choice(non_overlaped_regions)
                    target = 'no'
                else:
                    region = random.choice(overlaped_regions)
                    target = 'yes'
                
                
                out_dict = {'unique_id': unique_id,
                            'image_source': image_source,
                            'task_name':  self.task_type,
                            'image_path': image_path,
                            'region': [region],
                            'options':['yes','no'],
                            'target_txt': target,
                            'meta_data': {'object_regions':{'given_region':[given_region]}}
                }
                fout.write(json.dumps(out_dict)+'\n')
                count+=1
                
                if count == self.num_train and data_type == 'train':
                    return
                elif count == self.num_val and data_type == 'valid':
                    return
                
    def create_data(self):

        meta_data = {
            "originial_data_dir": str(self.train_dir)
        }
        
        with open(self.out_data_dir / self.meta_info_fn, 'w') as f:
            json.dump(meta_data, f)


        region_descriptions = json.load(open(self.train_dir,'r'))
        train_data = region_descriptions[self.data_splits['train'][0]:self.data_splits['train'][1]]
        valid_data = region_descriptions[self.data_splits['valid'][0]:self.data_splits['valid'][1]]
        random.shuffle(train_data)
        random.shuffle(valid_data)
        
        if self.task_type == 'VG':
            self.create_VG_inst(train_data,data_type='train')
            self.create_VG_inst(valid_data,data_type='valid')
        elif self.task_type == 'VG_selection':
            self.create_VG_selection_inst(train_data,data_type='train')
            self.create_VG_selection_inst(valid_data,data_type='valid')
        elif self.task_type == 'GC':
            self.create_GC_inst(train_data,data_type='train')
            self.create_GC_inst(valid_data,data_type='valid')
        elif self.task_type == 'GC_selection':
            self.create_GC_selection_inst(train_data,data_type='train')
            self.create_GC_selection_inst(valid_data,data_type='valid')
        elif self.task_type == 'select_overlap_most_region':
            self.create_select_overlap_most_region_inst(train_data,data_type='train')
            self.create_select_overlap_most_region_inst(valid_data,data_type='valid')
        elif self.task_type == 'select_overlap_least_region':
            self.create_select_overlap_least_region_inst(train_data,data_type='train')
            self.create_select_overlap_least_region_inst(valid_data,data_type='valid')
        elif self.task_type == 'region_area':
            self.create_region_area_inst(train_data,data_type='train')
            self.create_region_area_inst(valid_data,data_type='valid')
        elif self.task_type == 'select_overlaped_region':
            self.create_select_overlaped_region_inst(train_data,data_type='train')
            self.create_select_overlaped_region_inst(valid_data,data_type='valid')
        elif self.task_type == 'select_nonoverlaped_region':
            self.create_select_nonoverlaped_region_inst(train_data,data_type='train')
            self.create_select_nonoverlaped_region_inst(valid_data,data_type='valid')
        elif self.task_type == 'if_region_overlap':
            self.create_if_region_overlap_inst(train_data,data_type='train')
            self.create_if_region_overlap_inst(valid_data,data_type='valid')
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