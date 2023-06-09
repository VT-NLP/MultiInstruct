"""given image generate caption"""
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

    def create_data(self):

        meta_data = {
            "originial_data_dir": str(self.train_dir)
        }
        
        with open(self.out_data_dir / self.meta_info_fn, 'w') as f:
            json.dump(meta_data, f)

        if self.task_type == 'image_caption':
            print('start loading coco2014 caption val data: ', self.val_dir)
            # first valid
            save_path = self.out_data_dir / 'valid.jsonl'
            input_dict = json.load(open(self.val_dir, 'r'))

            with open(save_path,'w') as fout:
                # images = input_dict['images']
                annotations = input_dict['annotations']
                print(f'total num of val {len(annotations)}, span num of val {self.num_val}')
                annotations = random.sample(annotations,self.num_val)
                for line in annotations:
                    image_path = os.path.join(self.val_img_dir,f"COCO_val2014_{line['image_id']:012d}.jpg")
                    # assert exists(image_path)
                    out_dict = {'unique_id': 'mscoco_caption2014_'+str(line['id']),
                                'image_source': 'coco2014',
                                'task_name':  self.task_type,
                                'image_path': os.path.join(self.val_img_dir,f"COCO_val2014_{line['image_id']:012d}.jpg"),
                                'target_txt': line['caption']
                    }
                    fout.write(json.dumps(out_dict)+'\n')
            print('end loading coco2014 caption val data: ',save_path)

            # train -----------------------------------------
            print('start loading coco2014 detection train data: ', self.train_dir)
            save_path = self.out_data_dir / 'train.jsonl'
            input_dict = json.load(open(self.train_dir, 'r'))

            with open(save_path,'w') as fout:
                annotations = input_dict['annotations']
                print(f'total num of train {len(annotations)}, span num of train {self.num_train}')
                if not self.num_train == -1:
                    annotations = random.sample(annotations,self.num_train)
                images_dict = []
                for line in annotations:
                    image_path = os.path.join(self.train_img_dir,f"COCO_train2014_{line['image_id']:012d}.jpg")
                    assert exists(image_path)
                    out_dict = {'unique_id': 'mscoco_caption2014_'+str(line['id']),
                                'image_source': 'coco2014',
                                'task_name':  self.task_type,
                                'image_path': os.path.join(self.train_img_dir,f"COCO_train2014_{line['image_id']:012d}.jpg"),
                                'target_txt': line['caption']
                    }
                    fout.write(json.dumps(out_dict)+'\n')
            print('end loading coco2014 caption train data: ', save_path)
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