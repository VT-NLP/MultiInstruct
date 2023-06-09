cd $RAW_DATASETS

echo "Downloading COCO Text..."

mkdir coco_text

cd coco_text

wget https://vision.cornell.edu/se3/wp-content/uploads/2019/05/COCO_Text.zip

unzip COCO_Text.zip && rm COCO_Text.zip