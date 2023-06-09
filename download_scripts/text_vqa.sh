cd $RAW_DATASETS

mkdir TextVQA
cd TextVQA

echo "Downloading Text-VQA..."

# train_val
mkdir train_val
cd train_val
wget https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip
wget https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_train.json
wget https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json
wget https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_Rosetta_OCR_v0.2_train.json

unzip train_val_images.zip && rm train_val_images.zip

# test
cd ..
mkdir test
cd test
wget https://dl.fbaipublicfiles.com/textvqa/images/test_images.zip
wget https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_test.json
wget https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_Rosetta_OCR_v0.2_test.json

unzip test_images.zip && rm test_images.zip