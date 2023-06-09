cd $RAW_DATASETS


mkdir visdial
cd visdial

echo "Downloading visdial..."

wget https://www.dropbox.com/s/ix8keeudqrd8hn8/visdial_1.0_train.zip?dl=0
wget https://www.dropbox.com/s/ibs3a0zhw74zisc/visdial_1.0_val.zip?dl=0
wget https://www.dropbox.com/s/twmtutniktom7tu/VisualDialog_val2018.zip?dl=0

unzip visdial_1.0_train.zip?dl=0 && rm visdial_1.0_train.zip?dl=0
unzip visdial_1.0_val.zip?dl=0 && rm visdial_1.0_val.zip?dl=0
unzip VisualDialog_val2018.zip?dl=0 && rm VisualDialog_val2018.zip?dl=0