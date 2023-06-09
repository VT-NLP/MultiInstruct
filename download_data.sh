#!/bin/bash
export RAW_DATASETS=raw_datasets

sh ./download_scripts/coco_text.sh
sh ./download_scripts/mscoco2014.sh
sh ./download_scripts/refcocog.sh
sh ./download_scripts/TDIUC.sh
sh ./download_scripts/visual7w.sh 
sh ./download_scripts/wikihow.sh
sh ./download_scripts/gqa.sh
sh ./download_scripts/nlvr.sh
sh ./download_scripts/refcoco.sh
sh ./download_scripts/text_vqa.sh
sh ./download_scripts/visual_spatial_reasoning.sh
sh ./download_scripts/medic.sh
sh ./download_scripts/ok_vqa.sh
sh ./download_scripts/refcoco+.sh
sh ./download_scripts/vaw.sh
sh ./download_scripts/vizwiz_image_quality.sh
sh ./download_scripts/mocheg.sh
sh ./download_scripts/snli_ve.sh
sh ./download_scripts/visdial.sh
sh ./download_scripts/vqa_v2.sh
