import os
import sys
import json
import glob
from instruction_templates import build_instruction
import pdb
from pathlib import Path
from os.path import exists
from util import process_region


OUTPUT_REGION_TASK = {
    "detection", "VG", "object_region_selection", "region_generation","pointing_grounded_VQA","descriptive_object_region_generate", "text_localization", "visual_object_region", "visual_subject_region", "descriptive_object_region_select", "VG_selection", "grounded_VQA", "select_overlap_most_region", "select_overlap_least_region", "select_overlaped_region", "select_nonoverlaped_region"
}

OUTPUT_IMAGE_CODE_TASK = {"image_generation", "infilling", "im_region_extraction", "im_descriptive_infilling", "image_completion",  "image_completion_w_image_caption", "image_completion_w_region_caption", "im_descriptive_extraction"}

NO_IMAGE_AS_INPUT = {'image_generation'}

OPTIONS_REGION_TASK = {
    "object_region_selection","pointing_grounded_VQA", "text_localization", "descriptive_object_region_select","VG_selection", "grounded_VQA", "select_overlap_most_region", "select_overlap_least_region","select_overlaped_region", "select_nonoverlaped_region"
}

META_REGION_TASK = {
    "visual_answer_justification", "commonsense_VQA", "visual_object_region", "visual_subject_region", "select_overlap_most_region", "select_overlap_least_region", "select_overlaped_region", "select_nonoverlaped_region", "if_region_overlap"
}

MISSING_TASK = {'VQA_absurd','region_area'}



output = []
with open('./train.jsonl','w') as fout:
    for file_name in glob.glob('training_data/*/*',recursive=True):
        assert '.json' in file_name
        if  not 'train.jsonl' in file_name:
            continue
        with open(file_name,'r') as fin:
            for line in fin:
                line = json.loads(line)
                image_path = line['image_path']
                image_path = os.path.abspath(image_path)
                # assert exists(image_path)
                task = line['task_name']
                
                if task in OUTPUT_IMAGE_CODE_TASK or task in MISSING_TASK: # next version
                    continue
                
                if task in OUTPUT_REGION_TASK:
                    # print(line)
                    line['region'] = [process_region(r) for r in line.get('region')]
                    line['target_txt'] = ' '.join(line['region'])
                elif line.get('region') is not None:
                    line['region'] = [process_region(r) for r in line.get('region')]
                if task in OPTIONS_REGION_TASK:
                    line['options'] = [process_region(r) for r in line.get('options')]
                if task in META_REGION_TASK:
                    for k in line['meta_data']['object_regions']:
                        meta_regions = line['meta_data']['object_regions'][k]
                        line['meta_data']['object_regions'][k] = [process_region(r) for r in meta_regions]
                # print(line)
                prompt, target = build_instruction(task, text=line.get('text'), options=line.get('options'), region=line.get('region'), context=line.get('context'), question=line.get('question'), explanation=line.get('explanation'), response=line.get('response'), premise=line.get('premise'), hypothesis=line.get('hypothesis'),answer=line.get('answer'), meta_data=line.get('meta_data'), target=line.get('target_txt'))
                line['prompt'] = prompt
                line['target'] = target
                fout.write(json.dumps(line)+'\n')
                break
                
with open('./test.jsonl','w') as fout:
    for file_name in glob.glob('testing_data/*/*',recursive=True):
        assert '.json' in file_name
        if  not 'test.jsonl' in file_name:
            continue
        with open(file_name,'r') as fin:
            for line in fin:
                line = json.loads(line)
                image_path = line['image_path']
                image_path = os.path.abspath(image_path)
                # assert exists(image_path)
                task = line['task_name']
                
                if task in OUTPUT_IMAGE_CODE_TASK or task in MISSING_TASK: # next version
                    continue
                print(line)
                if task in OUTPUT_REGION_TASK:
                    # print(line)
                    line['region'] = [process_region(r) for r in line.get('region')]
                    line['target_txt'] = ' '.join(line['region'])
                elif line.get('region') is not None:
                    line['region'] = [process_region(r) for r in line.get('region')]
                if task in OPTIONS_REGION_TASK:
                    line['options'] = [process_region(r) for r in line.get('options')]
                if task in META_REGION_TASK:
                    for k in line['meta_data']['object_regions']:
                        meta_regions = line['meta_data']['object_regions'][k]
                        line['meta_data']['object_regions'][k] = [process_region(meta_regions)]
                        # pdb.set_trace()
                # print(line)
                prompt, target = build_instruction(task, text=line.get('text'), options=line.get('options'), region=line.get('region'), context=line.get('context'), question=line.get('question'), explanation=line.get('explanation'), response=line.get('response'), premise=line.get('premise'), hypothesis=line.get('hypothesis'),answer=line.get('answer'), meta_data=line.get('meta_data'), target=line.get('target_txt'))
                line['prompt'] = prompt
                line['target'] = target
                fout.write(json.dumps(line)+'\n')
                break

        




