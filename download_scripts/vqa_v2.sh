
cd $RAW_DATASETS

echo "Downloading VQA..."

mkdir VQA_V2
cd VQA_V2

wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip

# unzip
unzip v2_Annotations_Train_mscoco.zip && rm v2_Annotations_Train_mscoco.zip
unzip v2_Annotations_Val_mscoco.zip && rm v2_Annotations_Val_mscoco.zip
unzip v2_Questions_Train_mscoco.zip && rm v2_Questions_Train_mscoco.zip
unzip v2_Questions_Val_mscoco.zip && rm v2_Questions_Val_mscoco.zip
unzip v2_Questions_Test_mscoco.zip && rm v2_Questions_Test_mscoco.zip

echo "Downloading COCO2014 images for VQA..."

wget http://images.cocodataset.org/zips/test2015.zip

# unzip
unzip test2015.zip && rm test2015.zip