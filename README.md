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

tqdm,torch==1.6.0,transformers==3.2.0

`pip install -r requirements.txt`

## Prepare Datasets


For data processing, we use moving sliding window with overlap to construct the context.

Follow these steps to obtain training data.

1. Unzip ace2004.zip and ace2005.zip into data/raw_data
2. Preprocessing the data with `preprocess.py`, the example used can be found in `scripts/prepare_ace2005.sh`

## Pretrained Model

We use [BERT-Base-Uncased](https://huggingface.co/bert-base-uncased)

## Train

Use `train.py`, following the example of `scripts/train_ace2005.sh`

## Evaluate

### Evaluation during training

In `train.py`, there are two parameters for evaluation: `--eval` and `--test_eval`

`--eval`: Evaluate Multi-turn QAs under gold entities as  universal NER task. The evaluated data is in the same format as the training data. If the sliding windows overlap, the results may be biased.

`--test_eval`: First evaluate the first turn of QA, and then construct the second turn of question and answer data based on the results of the first turn of QA to evaluate the second turn of QA. It will eliminate duplicate predictions caused by overlapping sliding windows.

### Evaluate the saved model:

Use `ckpt_eval.py` to evaluate the saved model.