HOME=/home/wangnan
REPO=$HOME/ERE-MQA
PRETRAINED_MODEL_PATH=$HOME/pretrained_models/bert-base-uncased

#training data
python $REPO/preprocess.py \
--data_dir $REPO/data/raw_data/ACE2004/train0 \
--dataset_tag ace2004 \
--window_size 300 \
--overlap 15 \
--threshold 1 \
--max_distance 45 \
--output_base_dir $REPO/data/cleaned_data/ACE2004 \
--pretrained_model_path  $PRETRAINED_MODEL_PATH 

#test data
python $REPO/preprocess.py \
--data_dir $REPO/data/raw_data/ACE2004/test0 \
--dataset_tag ace2004 \
--output_base_dir $REPO/data/cleaned_data/ACE2004 \
--pretrained_model_path  $PRETRAINED_MODEL_PATH \
--is_test