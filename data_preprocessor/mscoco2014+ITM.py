"""
given image and caption decide if they match
"""
from argparse import ArgumentParser
import json
from pathlib import Path
import os
import random
from os.path import exists

class InstructData:

    def __init__(self, args):
        self.train_dir = './raw_datasets/MSCOCO2014/annotations/captions_train2014.json'
        self.val_dir = './raw_datasets/MSCOCO2014/annotations/captions_val2014.json'
        self.train_img_dir = './raw_datasets/MSCOCO2014/train2014'
        self.val_img_dir = './raw_datasets/MSCOCO2014/val2014'
        self.task_type = args.task_type
        self.out_data_dir = args.out_data_dir
        self.out_data_dir.mkdir(parents=True, exist_ok=True)
        self.meta_info_fn = 'meta.json'
        self.num_train = args.num_train
        self.num_val = args.num_val
        self.vocabs = [['Yes','No'],['Yes, the text matches the content of the image','No, the text does not match the content of the image'],['the description matches the image','the text is not a description of the image'],['True','False'],['match','not match'] ]
    
    def create_ITM_inst(self, save_path, input_dict, num_inst, split):
        with open(save_path,'w') as fout:
            # images = input_dict['images']
            annotations = input_dict['annotations']
            print(f'total num of {split} {len(annotations)}, span num of {split} {num_inst}')
            if not num_inst == -1:
                annotations = random.sample(annotations,num_inst)
            for line in annotations:
                if split == 'valid':  
                    image_path = os.path.join(self.val_img_dir,f"COCO_val2014_{line['image_id']:012d}.jpg")
                else:
                    image_path = os.path.join(self.train_img_dir,f"COCO_train2014_{line['image_id']:012d}.jpg")
                assert exists(image_path)
                
                vocabs = random.choice(self.vocabs)
                if random.uniform(0,1)>0.5:
                    caption = line['caption']
                    target_txt = vocabs[0]
                else:
                    # random select a caption
                    while True:
                        line2 = random.choice(annotations)
                        if not line2['image_id'] == line['image_id']:
                            caption = line2['caption']
                            break
                    target_txt = vocabs[1]
                out_dict = {'unique_id': 'mscoco_caption2014_'+str(line['id']),
                                'image_source': 'coco2014',
                                'task_name':  self.task_type,
                                'image_path': image_path,
                                'text': caption,
                                'target_txt': target_txt,
                                'options':vocabs
                    }
                            
                fout.write(json.dumps(out_dict)+'\n')
        print(f'end loading coco2014 {split} data: ',save_path)
        
    def create_image_text_selection_inst(self, save_path, input_dict, num_inst, split):
        with open(save_path,'w') as fout:
            # images = input_dict['images']
            annotations = input_dict['annotations']
            print(f'total num of {split} {len(annotations)}, span num of {split} {num_inst}')
            if not num_inst == -1:
                annotations = random.sample(annotations,num_inst)
            for line in annotations:
                if split == 'valid':  
                    image_path = os.path.join(self.val_img_dir,f"COCO_val2014_{line['image_id']:012d}.jpg")
                else:
                    image_path = os.path.join(self.train_img_dir,f"COCO_train2014_{line['image_id']:012d}.jpg")
                assert exists(image_path)
                
                caption = line['caption']
                # random select a caption
                
                options = []
                while len(options) < 4:
                    line2 = random.choice(annotations)
                    if not line2['image_id'] == line['image_id']:
                        options.append(line2['caption'])
                options.append(caption)
                random.shuffle(options)
                out_dict = {'unique_id': 'mscoco_caption2014_'+str(line['id']),
                                'image_source': 'coco2014',
                                'task_name':  self.task_type,
                                'image_path': image_path,
                                'target_txt': caption,
                                'options':options
                    }
                            
                fout.write(json.dumps(out_dict)+'\n')
        print(f'end loading coco2014 {split} data: ',save_path)
    
    
    def create_data(self):

        meta_data = {
            "originial_data_dir": str(self.train_dir)
        }
        
        with open(self.out_data_dir / self.meta_info_fn, 'w') as f:
            json.dump(meta_data, f)

        if self.task_type == 'ITM':
            
            save_path = self.out_data_dir / 'valid.jsonl'
            input_dict = json.load(open(self.val_dir, 'r'))
            self.create_ITM_inst(save_path, input_dict, self.num_val, 'valid')
            
            save_path = self.out_data_dir / 'train.jsonl'
            input_dict = json.load(open(self.train_dir, 'r'))
            self.create_ITM_inst(save_path, input_dict, self.num_train, 'train')
            
        elif self.task_type == 'image_text_selection':
            
            save_path = self.out_data_dir / 'valid.jsonl'
            input_dict = json.load(open(self.val_dir, 'r'))
            self.create_image_text_selection_inst(save_path, input_dict, self.num_val, 'valid')
            
            save_path = self.out_data_dir / 'train.jsonl'
            input_dict = json.load(open(self.train_dir, 'r'))
            self.create_image_text_selection_inst(save_path, input_dict, self.num_train, 'train')

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