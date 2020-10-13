HOME=/home/wangnan
REPO=$HOME/ERE-MQA
PRETRAINED_MODEL=/home/wangnan/pretrained_models/bert-base-uncased


python $REPO/train.py \
--dataset_tag ace2004 \
--train_path $REPO/data/cleaned_data/ACE2004/bert-base-uncased_overlap_15_window_300_threshold_1_max_distance_45/train0.json \
--train_batch 20 \
--test_path $REPO/data/cleaned_data/ACE2004/test0.json \
--test_batch 20 \
--pretrained_model_path $PRETRAINED_MODEL \
--max_epochs 10 \
--warmup_ratio 0.1 \
--lr 2e-5 \
--theta 0.25 \
--window_size 300 \
--overlap 45 \
--threshold 3 \
--max_grad_norm 1 \
--test_eval \
--seed 0 \
--amp