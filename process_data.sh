#!/bin/bash
mkdir training_data
mkdir testing_data

max_num_train=$1
max_num_dev=$2
max_num_test=$3

python data_preprocessor/mscoco2014+caption.py --out_data_dir training_data/mscoco2014+image_caption --task_type image_caption --num_train ${max_num_train} --num_val ${max_num_dev}

python data_preprocessor/mscoco2014+object_grounding.py --out_data_dir training_data/mscoco2014+object_grounding  --task_type object_grounding --num_train ${max_num_train} --num_val ${max_num_dev}

python data_preprocessor/mscoco2014+region_object_match.py --out_data_dir training_data/mscoco2014+object_region_match  --task_type object_region_match --num_train ${max_num_train} --num_val ${max_num_dev}

python data_preprocessor/mscoco2014+object_match.py --out_data_dir training_data/mscoco2014+object_match  --task_type object_match --num_train ${max_num_train} --num_val ${max_num_dev}

python data_preprocessor/mscoco2014+object_region_selection.py --out_data_dir training_data/mscoco+object_region_selection  --task_type object_region_selection --num_train ${max_num_train} --num_val ${max_num_dev}

python data_preprocessor/mscoco2014+missing_object_selection.py --out_data_dir training_data/mscoco2014+missing_object_selection  --task_type missing_object_selection --num_train ${max_num_train} --num_val ${max_num_dev}

python data_preprocessor/mscoco2014+ITM.py --out_data_dir training_data/mscoco2014+ITM --task_type ITM --num_train ${max_num_train} --num_val ${max_num_dev}

python data_preprocessor/mscoco2014+region_object_selection.py --out_data_dir training_data/mscoco2014+region_object_selection  --task_type region_object_selection --num_train ${max_num_train} --num_val ${max_num_dev}

python data_preprocessor/mscoco2014+region_generation.py --out_data_dir training_data/mscoco2014+region_generation  --task_type region_generation --num_train ${max_num_train} --num_val ${max_num_dev}

python data_preprocessor/mscoco2014+ITM.py --out_data_dir training_data/mscoco2014+image_text_selection --task_type image_text_selection  --num_train ${max_num_train} --num_val ${max_num_dev}

python data_preprocessor/mscoco_text.py --out_data_dir training_data/mscoco_text+text_localization --task_type text_localization --num_train ${max_num_train} --num_val ${max_num_dev}

python data_preprocessor/mscoco_text.py --out_data_dir training_data/mscoco_text+text_legibility --task_type text_legibility --num_train ${max_num_train} --num_val ${max_num_dev}

python data_preprocessor/mscoco_text.py --out_data_dir training_data/mscoco_text+text_type --task_type text_type --num_train ${max_num_train} --num_val ${max_num_dev}

python data_preprocessor/mscoco_text.py --out_data_dir training_data/mscoco_text+region_text_match --task_type region_text_match --num_train ${max_num_train} --num_val ${max_num_dev}

python data_preprocessor/refcoco+descriptive_object_identification.py --out_data_dir training_data/refcoco+descriptive_object_region_generate --task_type descriptive_object_region_generate --num_train ${max_num_train} --num_val ${max_num_dev}

python data_preprocessor/refcoco+descriptive_object_identification.py --out_data_dir training_data/refcoco+descriptive_object_region_select --task_type descriptive_object_region_select --num_train ${max_num_train} --num_val ${max_num_dev}

python data_preprocessor/refcoco+descriptive_object_identification.py --out_data_dir training_data/refcoco+object_description_generate --task_type object_description_generate --num_train ${max_num_train} --num_val ${max_num_dev}

python data_preprocessor/VQAv2+open-domain_VQA.py --out_data_dir training_data/VQAv2+open_domain_VQA --task_type open-domain_VQA --num_train ${max_num_train} --num_val ${max_num_dev}

python data_preprocessor/VQAv2+question_image_match.py --out_data_dir training_data/VQAv2+question_image_match --task_type question_image_match --num_train ${max_num_train} --num_val ${max_num_dev}

python data_preprocessor/vizwiz+image_quality.py --out_data_dir training_data/vizwiz+image_quality --task_type image_quality --num_train ${max_num_train} --num_val ${max_num_dev}

# python data_preprocessor/wikihow.py --out_data_dir training_data/wikihow+next_step --task_type wikihow_next_step --num_train ${max_num_train} --num_val ${max_num_dev}

# python data_preprocessor/wikihow.py --out_data_dir training_data/wikihow+text_image_step_order --task_type wikihow_text_image_step_order --num_train ${max_num_train} --num_val ${max_num_dev}

# python data_preprocessor/wikihow.py --out_data_dir training_data/wikihow+image_text_step_order --task_type wikihow_image_text_step_order --num_train ${max_num_train} --num_val ${max_num_dev}

# python data_preprocessor/wikihow.py --out_data_dir training_data/wikihow+immediate_next_step_selection --task_type wikihow_immediate_next_step_selection --num_train ${max_num_train} --num_val ${max_num_dev}

python data_preprocessor/mocheg.py --out_data_dir training_data/mocheg+multimodal_factual_checking --task_type multimodal_factual_checking --num_train ${max_num_train} --num_val ${max_num_dev}

python data_preprocessor/TDIUC.py --num_train ${max_num_train} --num_val ${max_num_dev}

python data_preprocessor/VAW+visual_attribute.py --out_data_dir training_data/VAW+visual_attribute --task_type visual_attribute --num_train ${max_num_train} --num_val ${max_num_dev}

python data_preprocessor/visual7w_VQA.py --out_data_dir training_data/visual7w+VQA --task_type VQA  --num_train ${max_num_train} --num_valid ${max_num_dev}

python data_preprocessor/GQA.py --out_data_dir training_data/GQA+open-domain_VQA --task_type open-domain_VQA --num_train ${max_num_train} --num_val ${max_num_dev}

python data_preprocessor/GQA.py --out_data_dir training_data/GQA+open-domain_VQA --task_type open-domain_VQA --num_train ${max_num_train} --num_val ${max_num_dev}

python data_preprocessor/OKVQA.py --out_data_dir training_data/OKVQA+open-domain_VQA --num_train ${max_num_train} --num_val ${max_num_dev}

python data_preprocessor/visualgenome+open-domain_VQA.py --out_data_dir training_data/visualgenome+open-domain_VQA --task_type open-domain_VQA --num_train ${max_num_train} --num_val ${max_num_dev}

python data_preprocessor/visualgenome+region_caption.py --out_data_dir training_data/visualgenome+grounded_caption --task_type GC --num_train ${max_num_train} --num_val ${max_num_dev}

python data_preprocessor/visualgenome+region_caption.py --out_data_dir training_data/visualgenome+visual_grounding --task_type VG --num_train ${max_num_train} --num_val ${max_num_dev}

python data_preprocessor/visualgenome+region_caption.py --out_data_dir training_data/visualgenome+visual_grounding_selection --task_type VG_selection --num_train ${max_num_train} --num_val ${max_num_dev}

python data_preprocessor/visualgenome+object_relationship.py --out_data_dir training_data/visualgenome+object_relationship --task_type object_relationship --num_train ${max_num_train} --num_val ${max_num_dev}

python data_preprocessor/visualgenome+object_relationship.py --out_data_dir training_data/visualgenome+visual_object_identification --task_type visual_object_identification --num_train ${max_num_train} --num_val ${max_num_dev}

python data_preprocessor/visualgenome+object_relationship.py --out_data_dir training_data/visualgenome+visual_subject_identification --task_type visual_subject_identification --num_train ${max_num_train} --num_val ${max_num_dev}

python data_preprocessor/visualgenome+object_relationship.py --out_data_dir training_data/visualgenome+visual_object_region --task_type visual_object_region --num_train ${max_num_train} --num_val ${max_num_dev}

python data_preprocessor/visualgenome+object_relationship.py --out_data_dir training_data/visualgenome+visual_subject_region --task_type visual_subject_region --num_train ${max_num_train} --num_val ${max_num_dev}

# ---------------------------------------------------------------------------------

# ------------------------------- region tasks ---------------------------
python data_preprocessor/visualgenome+region_caption.py --out_data_dir training_data/visualgenome+select_overlap_most_region --task_type select_overlap_most_region  --num_train ${max_num_train} --num_val ${max_num_dev}

python data_preprocessor/visualgenome+region_caption.py --out_data_dir training_data/visualgenome+region_area --task_type region_area  --num_train ${max_num_train} --num_val ${max_num_dev}

python data_preprocessor/visualgenome+region_caption.py --out_data_dir training_data/visualgenome+select_overlaped_region --task_type select_overlaped_region   --num_train ${max_num_train} --num_val ${max_num_dev}

python data_preprocessor/visualgenome+region_caption.py --out_data_dir training_data/visualgenome+select_overlap_least_region --task_type select_overlap_least_region  --num_train ${max_num_train} --num_val ${max_num_dev}

python data_preprocessor/visualgenome+region_caption.py --out_data_dir training_data/visualgenome+if_region_overlap --task_type if_region_overlap   --num_train ${max_num_train} --num_val ${max_num_dev}

python data_preprocessor/visualgenome+region_caption.py --out_data_dir training_data/visualgenome+select_nonoverlaped_region --task_type select_nonoverlaped_region   --num_train ${max_num_train} --num_val ${max_num_dev}

#  --------- testing tasks ----------

python data_preprocessor/nlvr.py --out_data_dir testing_data/nlvr+natural_language_visual_reasoning --task_type natural_language_visual_reasoning

python data_preprocessor/visual_spatial_reasoning.py --out_data_dir testing_data/visual_spatial_reasoning --task_type visual_spatial_reasoning

python data_preprocessor/snli_ve.py --out_data_dir testing_data/visual_nli --task_type visual_nli --num_test -1

python data_preprocessor/text_vqa.py --out_data_dir testing_data/text_vqa --task_type text_vqa --num_test -1

python data_preprocessor/vcr.py --out_data_dir testing_data/commonsense_VQA --task_type commonsense_VQA --num_test -1

python data_preprocessor/visdial.py --out_data_dir testing_data/visual_dialog --task_type visual_dialog  --num_test -1

python data_preprocessor/hateful_memes.py --out_data_dir testing_data/visual_text_extraction --task_type visual_text_extraction  --num_test -1

python data_preprocessor/visual7W.py --out_data_dir testing_data/grounded_VQA --task_type grounded_VQA  --num_test -1

python data_preprocessor/medic.py --out_data_dir testing_data  --num_test -1