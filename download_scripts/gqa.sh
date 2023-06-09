cd $RAW_DATASETS


mkdir GQA
cd GQA

echo "Downloading GQA..."

wget https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip
wget https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip

unzip questions1.2.zip && rm questions1.2.zip
unzip images.zip && rm images.zip