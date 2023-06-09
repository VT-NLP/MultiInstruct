
cd $RAW_DATASETS

echo "Downloading refcocog..."

mkdir refcocog
cd refcocog

wget https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip

unzip refcocog.zip && rm refcocog.zip