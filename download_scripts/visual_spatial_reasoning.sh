
cd $RAW_DATASETS

echo "Downloading visual spatial reasoning..."

mkdir visual_spatial_reasoning
cd visual_spatial_reasoning

git clone https://github.com/cambridgeltl/visual-spatial-reasoning.git

cd visual-spatial-reasoning/data

echo "Downloading COCO-2017 for visual spatial reasoning..."

wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
unzip train2017.zip && unzip val2017.zip
mv train2017 trainval2017 && mv val2017/* trainval2017 && rm -r val2017

mkdir images
python select_only_revlevant_images.py data_files/all_vsr_validated_data.jsonl/  trainval2017/ images/