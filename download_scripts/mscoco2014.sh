cd $RAW_DATASETS

mkdir MSCOCO2014
cd MSCOCO2014

echo "Downloading MSCOCO2014..."

# train
wget http://images.cocodataset.org/zips/train2014.zip

unzip train2014.zip && rm train2014.zip

# validation
wget http://images.cocodataset.org/zips/val2014.zip
unzip val2014.zip && rm val2014.zip

wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
unzip annotations_trainval2014.zip && rm annotations_trainval2014.zip