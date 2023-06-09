

cd $RAW_DATASETS

mkdir TDIUC
cd TDIUC

echo "Downloading TDIUC..."

wget https://kushalkafle.com/data/TDIUC.zip

unzip TDIUC.zip && rm TDIUC.zip
