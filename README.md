# Entity-Relation Extraction as Multi-Turn Question Answering
The repository contains the code of the recent research advances in [Shannon.AI](http://www.shannonai.com). 

**Entity-Relation Extraction as Multi-turn Question Answering (ACL 2019)**<br>
Xiaoya Li, Fan Yin, Zijun Sun, Xiayu Li, Arianna Yuan, Duo Chai, Mingxin Zhou and Jiwei Li <br> 
Accepted by [ACL 2019](https://arxiv.org/pdf/1905.05529.pdf) <br>
If you find this repo helpful, please cite the following:
```text
@inproceedings{li-etal-2019-entity,
    title = "Entity-Relation Extraction as Multi-Turn Question Answering",
    author = "Li, Xiaoya  and
      Yin, Fan  and
      Sun, Zijun  and
      Li, Xiayu  and
      Yuan, Arianna  and
      Chai, Duo  and
      Zhou, Mingxin  and
      Li, Jiwei",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/P19-1129",
    doi = "10.18653/v1/P19-1129",
    pages = "1340--1350"
}
```


## Install Requirements

`pip install -r requirements.txt`

## Prepare Data
This project currently only support ace2004 and ace2005 dataset.

For data processing:
1. We use moving sliding window with overlap to split the passage, 
2. We use a threshold to filter impossible relations with small frequency, 
3. We use max distance to further filter impossible relations that have end entity type that doesn't  in a small range from the head entity.
4. We use two special tokens to mark the head entity in the context to avoid coreference problem in the window. 

Following these steps to obtain training data:

1. Unzip ace2004.zip and ace2005.zip into data/raw_data
2. Preprocess the data with `preprocess.py`.The example used can be found in `scripts/prepare_ace2005.sh`

## Pretrained Model

We use [BERT-Base-Uncased](https://huggingface.co/bert-base-uncased)

## Train

Use `train.py` to train the model. 

The example used can be found in `scripts/train_ace2005.sh`.

## Evaluate checkpoints:

Use `ckpt_eval.py` to evaluate the saved model.

The example used can be found in `scripts/ckpt_eval.sh`.