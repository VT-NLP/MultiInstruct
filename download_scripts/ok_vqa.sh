
cd $RAW_DATASETS

mkdir OK-VQA
cd OK-VQA

echo "Downloading OK-VQA..."

# train
mkdir train
cd train
wget https://okvqa.allenai.org/static/data/mscoco_train2014_annotations.json.zip
wget https://okvqa.allenai.org/static/data/OpenEnded_mscoco_train2014_questions.json.zip

unzip mscoco_train2014_annotations.json.zip && rm mscoco_train2014_annotations.json.zip
unzip OpenEnded_mscoco_train2014_questions.json.zip && rm OpenEnded_mscoco_train2014_questions.json.zip

# validation
cd ..
mkdir val
cd val
wget https://okvqa.allenai.org/static/data/mscoco_val2014_annotations.json.zip
wget https://okvqa.allenai.org/static/data/OpenEnded_mscoco_val2014_questions.json.zip

unzip mscoco_val2014_annotations.json.zip && rm mscoco_val2014_annotations.json.zip
unzip OpenEnded_mscoco_val2014_questions.json.zip && rm OpenEnded_mscoco_val2014_questions.json.zip
