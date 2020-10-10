HOME=/home/wangnan
REPO=$HOME/ERE-MQA
python $REPO/train.py \
--dataset_tag ace2004 \
--train_path $REPO/data/cleaned_data/ACE2004/bert-base-uncased_overlap_15_window_300_threshold_5/train0.json \
--train_batch 20 \
--dev_path $REPO/data/cleaned_data/ACE2004/bert-base-uncased_overlap_15_window_300_threshold_5/test0.json \
--dev_batch 20 \
--test_path $REPO/data/cleaned_data/ACE2004/test0.json \
--test_batch 20 \
--pretrained_model_path /home/wangnan/pretrained_models/bert-base-uncased \
--max_epochs 10 \
--warmup_ratio 0.05 \
--lr 2e-5 \
--theta 0.25 \
--window_size 300 \
--overlap 45 \
--threshold 5 \
--max_grad_norm 1 \
--test_eval \
--amp