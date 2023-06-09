
cd $RAW_DATASETS

echo "Downloading vizwiz image quality..."

mkdir vizwiz_image_quality
cd vizwiz_image_quality

wget https://vizwiz.cs.colorado.edu/VizWiz_final/images/train.zip
wget https://vizwiz.cs.colorado.edu/VizWiz_final/images/val.zip
wget https://vizwiz.cs.colorado.edu/VizWiz_final/images/test.zip
wget https://vizwiz.cs.colorado.edu/VizWiz_final/image_quality/annotations.zip


unzip train.zip && rm train.zip
unzip val.zip && rm val.zip
unzip test.zip && rm test.zip
unzip annotations.zip && rm annotations.zip