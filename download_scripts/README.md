Below are the instructions for manually downloading some of the datasets that we can not automatically download for you.
## Hateful Memes
To acquire the data, you will need to register at DrivenData's Hateful Memes Competition and download data from the challenge's download page. You can find detailed instructions at https://github.com/facebookresearch/mmf/tree/main/projects/hateful_memes
```
cd raw_datasets
mkdir hateful_memes
cd hateful_memes
```
Download the dataset! The dataset should contain "dev_seen.jsonl" "dev_unseen.jsonl" "img" "LICENSE.txt" "README.md" "test_seen.jsonl"  "test_unseen.jsonl" "train.jsonl". 


## Visual Commonsense Reasoning
You can download the dataset at https://visualcommonsense.com/download/
```
cd raw_datasets
mkdir VCR
cd VCR
```
Download the dataset!
```
unzip vcr1images.zip && rm vcr1images.zip
unzip vcr1annots.zip && rm vcr1annots.zip
```

## VisDial Dataset
If VisDial Dataset is not successfully downloaded, you can download the dataset at https://visualdialog.org/data
```
cd raw_datasets
mkdir visdial
cd visdial
```
Download the dataset!
```
unzip visdial_1.0_train.zip?dl=0 && rm visdial_1.0_train.zip?dl=0
unzip visdial_1.0_val.zip?dl=0 && rm visdial_1.0_val.zip?dl=0
unzip VisualDialog_val2018.zip?dl=0 && rm VisualDialog_val2018.zip?dl=0
```

## WikiHow
If WikiHow Dataset is not successfully downloaded, You can download the dataset at https://drive.google.com/file/d/1vnDduJmuFpeT8yTgtBR9Bj8bXlI4zIJR/view?usp=share_link
```
cd raw_datasets
mkdir wikihow
cd wikihow
```
Download the dataset!
```
tar -xf wikihow.tar.gz && rm wikihow.tar.gz
```
## Visual Genome
```
cd raw_datasets
mkdir visual_genome
cd visual_genome
```
Download the dataset!
```
unzip images.zip && rm images.zip
unzip images2.zip && rm images2.zip
unzip image_data.json.zip && rm image_data.json.zip
unzip objects.json.zip && rm objects.json.zip
unzip question_answers.json.zip && rm question_answers.json.zip
unzip region_descriptions.json.zip && rm region_descriptions.json.zip
```
