cd $RAW_DATASETS

echo "Downloading MEDIC..."

mkdir MEDIC
cd MEDIC

wget https://crisisnlp.qcri.org/data/medic/MEDIC.tar.gz

tar -xzf MEDIC.tar.gz && rm MEDIC.tar.gz