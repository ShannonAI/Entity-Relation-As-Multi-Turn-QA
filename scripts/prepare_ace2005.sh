HOME=/home/wangnan
REPO=$HOME/ERE-MQA
PRETRAINED_MODEL_PATH=$HOME/pretrained_models/bert-base-uncased


#training data
python $REPO/preprocess.py \
--data_dir $REPO/data/raw_data/ACE2005/train \
--dataset_tag ace2005 \
--window_size 300 \
--overlap 15 \
--threshold 1 \
--max_distance 45 \
--output_base_dir $REPO/data/cleaned_data/ACE2005 \
--pretrained_model_path  $PRETRAINED_MODEL_PATH 

#development data
python $REPO/preprocess.py \
--data_dir $REPO/data/raw_data/ACE2005/dev \
--dataset_tag ace2005 \
--output_base_dir $REPO/data/cleaned_data/ACE2005 \
--pretrained_model_path  $PRETRAINED_MODEL_PATH \
--is_test

#test data
python $REPO/preprocess.py \
--data_dir $REPO/data/raw_data/ACE2005/test \
--dataset_tag ace2005 \
--output_base_dir $REPO/data/cleaned_data/ACE2005 \
--pretrained_model_path  $PRETRAINED_MODEL_PATH \
--is_test