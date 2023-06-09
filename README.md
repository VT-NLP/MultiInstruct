## MULTIINSTRUCT: Improving Multi-Modal Zero-Shot Learning via Instruction Tuning

This is the official repo for our ACL 2023 MULTIINSTRUCT [paper](https://arxiv.org/pdf/2212.10773.pdf). MULTIINSTRUCT is the first multimodal instruction tuning benchmark dataset that consists of 62 diverse multimodal tasks in a unified seq-toseq format covering 10 broad categories. The tasks are derived from 21 existing open-source datasets and each task is equipped with 5 expertwritten instructions.

<img src="multi_instruct_tasks_fig.png">

## What's New: 🎉 
  * [Call for Datasets] Call for new datasets <br>
  > If the datasets that you want to use are not in MultiInstruct, please let us know by raising issues. We will help you to write downloading scripts, preprocessing scripts and instructions and include the datasets in the repo.

  > If you have interesting datasets and you want us to include in MultiInstruct, please let us know by raising issues. We will help you to write downloading scripts, preprocessing scripts, and instructions and include the datasets in the repo.
  * [Dataset Release] July 2023, will release **MultiInstruct 1.5** <br>
  > In 1.5 version, we plan to relsease around 150 more multimodal tasks in addtion to the original 62 tasks to facilitate research in vision-language foundation models. The new tasks are all vision-language tasks in a unified seq-toseq format.
  * [Dataset Release] June 2023, released **MultiInstruct 1.0** <br>
  > First multimodal instruction tuning benchmark dataset consistens of 62 diverse multimodal tasks. 1.0 version is the version used in our paper.
### Downloading Dataset:
```
sh download_data.sh
```
This script helps you to automatically download the most of the datasets. For some of the datasets, due to the license issue we can not help you to automatically download them but you can manually download them by following the instructions [here](download_scripts).

### Preprocessing Dataset:
```
sh process_data.sh 1000 100
```
This script helps you to process all the datasets. The number "10000" is the max number of training instances for each task and "100" is the max number of validation instances for each task. You can change them based on your requirement. The procssed data is stored in training_data and testing_data directory.

### Assembling Dataset:
```
python instruction_templates.py
```
This script helps you to assemble the instances randomly with 1 of their 5 instructions and the generate the training and testing dataset (train.jsonl and test.jsonl). An example instance in the dataset looks like
```
{"unique_id": "mscoco_text_legibility_287140", "image_source": "coco2014", "task_name": "text_legibility", "image_path": "raw_datasets/MSCOCO2014/train2014/COCO_train2014_000000287140.jpg", "region": ["212.35294117647058 108.67088607594941 238.16806722689077 125.88607594936713"], "options": ["not clear and complete", "clear"], "target_txt": "clear", "prompt": "Decide if the text in the given region is legible. Region 212.35294117647058 108.67088607594941 238.16806722689077 125.88607594936713 \n\n[Options]: not clear and complete||||clear", "target": "clear"}
```
You can load the image via "image_path" and your input prompt is in "prompt" field. The output is in "target" field. You can ignore all other fileds in each instance.
## Citation 
If you're using MultiInstruct in your research or applications, please cite using this BibTeX:
```bibtex
@misc{xu2023multiinstruct,
      title={MultiInstruct: Improving Multi-Modal Zero-Shot Learning via Instruction Tuning}, 
      author={Zhiyang Xu and Ying Shen and Lifu Huang},
      year={2023},
      eprint={2212.10773},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
## Question
If you face any problem when processing the dataset, please don't hesitate to contact us at zhiyangx@vt.edu. We will get back to you as soon as possible.
## License
Please carefully check the licenses for all the datasets in MultiInstruct on their official websites.
