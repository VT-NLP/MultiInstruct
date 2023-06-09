
cd $RAW_DATASETS

echo "Downloading conceptual visual7w..."

mkdir visual7w
cd visual7w

wget http://vision.stanford.edu/yukezhu/visual7w_images.zip
wget https://ai.stanford.edu/~yukez/papers/resources/dataset_v7w_telling.zip
wget https://ai.stanford.edu/~yukez/papers/resources/dataset_v7w_pointing.zip
wget https://ai.stanford.edu/~yukez/papers/resources/dataset_v7w_grounding_annotations.zip

unzip visual7w_images.zip && rm visual7w_images.zip
unzip dataset_v7w_telling.zip && rm dataset_v7w_telling.zip
unzip dataset_v7w_pointing.zip && rm dataset_v7w_pointing.zip
unzip dataset_v7w_grounding_annotations.zip && rm dataset_v7w_grounding_annotations.zip