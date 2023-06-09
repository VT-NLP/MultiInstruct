
cd $RAW_DATASETS


mkdir mocheg
cd mocheg

echo "Downloading mocheg..."

wget http://nlplab1.cs.vt.edu/~menglong/project/multimodal/fact_checking/MOCHEG/dataset/mocheg.tar.gz

tar -xzf mocheg.tar.gz && rm mocheg.tar.gz