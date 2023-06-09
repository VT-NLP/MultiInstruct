
cd $RAW_DATASETS

echo "Downloading refcoco..."

mkdir refcoco
cd refcoco

wget https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip

unzip refcoco.zip && rm refcoco.zip