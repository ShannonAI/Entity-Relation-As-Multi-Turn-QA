import torch
import numpy as np
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, DistributedSampler
from transformers import BertTokenizer

from preprocess import passage_blocks, get_question
from constants import *


def collate_fn(batch):
    # for training
    nbatch = {}
    for b in batch:
        for k, v in b.items():
            nbatch[k] = nbatch.get(k, []) + [torch.tensor(v)]
    txt_ids = nbatch['txt_ids']
    tags = nbatch['tags']
    context_mask = nbatch['context_mask']
    token_type_ids = nbatch['token_type_ids']
    turn_mask = nbatch['turn_mask']
    ntxt_ids = pad_sequence(txt_ids, batch_first=True,
                            padding_value=0)
    ntags = pad_sequence(tags, batch_first=True, padding_value=-1)
    ncontext_mask = pad_sequence(
        context_mask, batch_first=True, padding_value=0)
    ntoken_type_ids = pad_sequence(
        token_type_ids, batch_first=True, padding_value=1)
    attention_mask = torch.zeros(ntxt_ids.shape)
    for i in range(len(ntxt_ids)):
        txt_len = len(txt_ids[i])
        attention_mask[i, :txt_len] = 1
    nbatch['txt_ids'] = ntxt_ids
    nbatch['tags'] = ntags
    nbatch['context_mask'] = ncontext_mask
    nbatch['token_type_ids'] = ntoken_type_ids
    nbatch['attention_mask'] = attention_mask
    nbatch['turn_mask'] = torch.tensor(turn_mask, dtype=torch.uint8)
    return nbatch


def collate_fn1(batch):
    # for testing
    nbatch = {}
    for b in batch:
        for k, v in b.items():
            nbatch[k] = nbatch.get(k, []) + [torch.tensor(v)]
    txt_ids = nbatch['txt_ids']
    context_mask = nbatch['context_mask']
    token_type_ids = nbatch['token_type_ids']
    ntxt_ids = pad_sequence(txt_ids, batch_first=True,
                            padding_value=0)
    ncontext_mask = pad_sequence(
        context_mask, batch_first=True, padding_value=0)
    ntoken_type_ids = pad_sequence(
        token_type_ids, batch_first=True, padding_value=1)
    attention_mask = torch.zeros(ntxt_ids.shape)
    for i in range(len(ntxt_ids)):
        txt_len = len(txt_ids[i])
        attention_mask[i, :txt_len] = 1
    nbatch['txt_ids'] = ntxt_ids
    nbatch['context_mask'] = ncontext_mask
    nbatch['token_type_ids'] = ntoken_type_ids
    nbatch['attention_mask'] = attention_mask
    return nbatch


def get_inputs(context, q, tokenizer, title="", max_len=200, ans=[], head_entity=None):
    query = tokenizer.tokenize(q)
    tags = [tag_idxs['O']]*len(context)
    for i, an in enumerate(ans):
        start, end = an[1:-1]
        end = end-1
        if start != end:
            tags[start] = tag_idxs['B']
            tags[end] = tag_idxs['E']
            for i in range(start+1, end):
                tags[i] = tag_idxs['M']
        else:
            tags[start] = tag_idxs['S']
    if head_entity:
        h_start, h_end = head_entity[1], head_entity[2]
        context = context[:h_start]+['[unused0]'] + \
            context[h_start:h_end]+["[unused1]"]+context[h_end:]
        assert len(context) == len(tags)+2
        tags = tags[:h_start]+[tag_idxs['O']] + \
            tags[h_start:h_end]+[tag_idxs['O']]+tags[h_end:]
    txt_len = len(query)+len(title)+len(context) + \
        4 if title else len(query)+len(context)+3
    if txt_len > max_len:
        context = context[:max_len -
                          len(query) - 3] if not title else context[:max_len-len(query)-len(title)-4]
        tags = tags[:max_len -
                    len(query) - 3] if not title else tags[:max_len-len(query)-len(title)-4]
    if title:
        txt = ['[CLS]'] + query+['[SEP]'] + \
            title + ['[SEP]'] + context + ['[SEP]']
    else:
        txt = ['[CLS]'] + query + ['[SEP]'] + context + ['[SEP]']
    txt_ids = tokenizer.convert_tokens_to_ids(txt)
    # [CLS] is used to judge whether there is an answe
    if not title:
        tags1 = [tag_idxs['O'] if len(
            ans) > 0 else tag_idxs['S']] + [-1] * (len(query) + 1) + tags + [-1]
        context_mask = [1] + [0] * (len(query) + 1) + [1] * len(context) + [0]
        token_type_ids = [0] * (len(query) + 2) + [1] * (len(context) + 1)
    else:
        tags1 = [tag_idxs['O'] if len(
            ans) > 0 else tag_idxs['S']] + [-1]*(len(query)+len(title)+2) + tags + [-1]
        context_mask = [1] + [0] * \
            (len(query)+len(title)+2) + [1] * len(context) + [0]
        token_type_ids = [0]*(len(query)+len(title)+3)+[1]*(len(context) + 1)
    return txt_ids, tags1, context_mask, token_type_ids


def query2relation(question, question_templates):
    '''
    query -> <entity_type,relation_type,entity_type>
    '''
    turn2_questions = question_templates['qa_turn2']
    turn2_questions = {v: k for k, v in turn2_questions.items()}
    for k, v in turn2_questions.items():
        k1 = k.replace("XXX.", "")
        if question.startswith(k1):
            return eval(v)
    raise Exception("cannot find the relation type corresponding to the query, if the \
                 query template is changed, please re-implement this function according to the new template")


class MyDataset:
    def __init__(self, dataset_tag, path, tokenizer, max_len=512, threshold=5):
        """
        Args:
            dataset_tag: type of dataset
            path:  path to training set file
            tokenizer： tokenizer of pretrained model
            max_len: max length of input
            threshold: only consider relationships where the frequency is greater than or equal to the threshold
        """
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.threshold = threshold
        self.dataset_tag = dataset_tag
        self.init_data()

    def init_data(self):
        self.all_t1 = []
        self.all_t2 = []
        if self.dataset_tag.lower() == "ace2004":
            idx1s = ace2004_idx1
            idx2s = ace2004_idx2
            dist = ace2004_dist
            question_templates = ace2004_question_templates
        elif self.dataset_tag.lower() == 'ace2005':
            idx1s = ace2005_idx1
            idx2s = ace2005_idx2
            dist = ace2005_dist
            question_templates = ace2005_question_templates
        else:
            raise Exception("this dataset is not yet supported")
        for d in tqdm(self.data, desc="dataset"):
            context = d['context']
            title = d['title']
            qa_pairs = d['qa_pairs']
            t1 = qa_pairs[0]
            t2 = qa_pairs[1]
            t1_qas = []
            t2_qas = []
            for i, (q, ans) in enumerate(t1.items()):
                txt_ids, tags, context_mask, token_type_ids = get_inputs(
                    context, q, self.tokenizer, title, self.max_len, ans)
                t1_qas.append(
                    {"txt_ids": txt_ids, "tags": tags, "context_mask": context_mask, "token_type_ids": token_type_ids, 'turn_mask': 0})
            for t in t2:
                head_entity = t['head_entity']
                for q, ans in t['qas'].items():
                    rel = query2relation(q, question_templates)
                    idx1, idx2 = rel[0], rel[1:]
                    idx1, idx2 = idx1s[idx1], idx2s[idx2]
                    if dist[idx1][idx2] >= self.threshold:
                        txt_ids, tags, context_mask, token_type_ids = get_inputs(
                            context, q, self.tokenizer, title, self.max_len, ans, head_entity)
                        t2_qas.append({"txt_ids": txt_ids, "tags": tags, "context_mask": context_mask,
                                       "token_type_ids": token_type_ids, 'turn_mask': 1})
            self.all_t1.extend(t1_qas)
            self.all_t2.extend(t2_qas)
        self.all_qas = self.all_t1+self.all_t2

    def __len__(self):
        return len(self.all_qas)

    def __getitem__(self, i):
        return self.all_qas[i]


class T1Dataset:
    def __init__(self, dataset_tag, test_path, tokenizer, window_size, overlap, max_len=512):
        """
        Args:
            dataset_tag: type of dataset
            test_path: path to test set file
            tokenizer: tokenizer of pretrained model
            window_size: sliding window size
            overlap: overlap between two adjacent windows
            max_len: max length of input
        """
        with open(test_path, encoding="utf=8") as f:
            data = json.load(f)
        self.dataset_tag = dataset_tag
        if dataset_tag.lower() == 'ace2004':
            dataset_entities = ace2004_entities
            question_templates = ace2004_question_templates
        elif dataset_tag.lower() == 'ace2005':
            dataset_entities = ace2005_entities
            question_templates = ace2005_question_templates
        else:
            raise Exception("this data set is not yet supported")
        self.t1_qas = []
        self.passages = []
        self.entities = []
        self.relations = []
        self.titles = []
        self.window_size = window_size
        self.overlap = overlap
        self.t1_querys = []
        self.t1_ids = []
        self.t1_gold = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        # passage_windows[i][j] represents the j-th window of the i-th passage
        self.passage_windows = []
        self.query_offset1 = []
        self.window_offset_base = window_size-overlap
        for ent_type in dataset_entities:
            query = get_question(question_templates, ent_type)
            self.t1_querys.append(query)
        for p_id, d in enumerate(tqdm(data, desc="t1_dataset")):
            passage = d["passage"]
            entities = d['entities']
            relations = d['relations']
            title = d['title']
            self.passages.append(passage)
            self.entities.append(entities)
            self.relations.append(relations)
            self.titles.append(title)
            blocks, _ = passage_blocks(passage, window_size, overlap)
            self.passage_windows.append(blocks)
            for ent in entities:
                self.t1_gold.append((p_id, tuple(ent[:-1])))
            for b_id, block in enumerate(blocks):
                for q_id, q in enumerate(self.t1_querys):
                    txt_ids, _, context_mask, token_type_ids = get_inputs(
                        block, q, tokenizer, title, max_len)
                    self.t1_qas.append({"txt_ids": txt_ids, "context_mask": context_mask,
                                        "token_type_ids": token_type_ids})
                    self.t1_ids.append((p_id, b_id, dataset_entities[q_id]))
                    ofs = len(title)+len(tokenizer.tokenize(q))+3
                    self.query_offset1.append(ofs)

    def __len__(self):
        return len(self.t1_qas)

    def __getitem__(self, i):
        return self.t1_qas[i]


class T2Dataset:
    def __init__(self, t1_dataset, t1_predict, threshold=5):
        '''
        Args:
            t1_dataset: an instance of T1Dataset
            t1_predict: predictions of the first turn QA
            threshold: only consider relationships where the frequency is greater than or equal to the threshold
        '''
        if t1_dataset.dataset_tag.lower() == "ace2004":
            idx1s = ace2004_idx1
            idx2s = ace2004_idx2
            dist = ace2004_dist
            dataset_entities = ace2004_entities
            dataset_relations = ace2004_relations
            question_templates = ace2004_question_templates
        elif t1_dataset.dataset_tag.lower() == 'ace2005':
            idx1s = ace2005_idx1
            idx2s = ace2005_idx2
            dist = ace2005_dist
            dataset_entities = ace2005_entities
            dataset_relations = ace2005_relations
            question_templates = ace2005_question_templates
        else:
            raise Exception("this data set is not yet supported")
        tokenizer = t1_dataset.tokenizer
        max_len = t1_dataset.max_len
        t1_ids = t1_dataset.t1_ids
        passages = t1_dataset.passages
        titles = t1_dataset.titles
        passage_windows = t1_dataset.passage_windows
        self.t2_qas = []
        self.t2_ids = []
        self.t2_gold = []
        self.query_offset2 = []
        relations = t1_dataset.relations
        entities = t1_dataset.entities
        query_offset1 = t1_dataset.query_offset1
        window_offset_base = t1_dataset.window_offset_base
        for passage_id, (ents, rels) in enumerate(zip(entities, relations)):
            for re in rels:
                head, rel, end = ents[re[1]], re[0], ents[re[2]]
                self.t2_gold.append(
                    (passage_id, (tuple(head[:-1]), rel, tuple(end[:-1]))))
        for i, (_id, pre) in enumerate(zip(tqdm(t1_ids, desc="t2 dataset"), t1_predict)):
            passage_id, window_id, head_entity_type = _id
            window_offset = window_offset_base*window_id
            context = passage_windows[passage_id][window_id]
            title = titles[passage_id]
            head_entities = []
            for start, end in pre:
                start1, end1 = start - \
                    query_offset1[i]+window_offset, end - \
                    query_offset1[i]+window_offset
                ent_str = tokenizer.convert_tokens_to_string(
                    passages[passage_id][start1:end1])
                head_entity = (head_entity_type, start1, end1, ent_str)
                head_entities.append(head_entity)
            for head_entity in head_entities:
                for rel in dataset_relations:
                    for end_ent_type in dataset_entities:
                        idx1, idx2 = idx1s[head_entity[0]
                                           ], idx2s[(rel, end_ent_type)]
                        if dist[idx1][idx2] >= threshold:
                            query = get_question(
                                question_templates, head_entity, rel, end_ent_type)
                            window_head_entity = (
                                head_entity[0], head_entity[1]-window_offset, head_entity[2]-window_offset, head_entity[3])
                            txt_ids, _, context_mask, token_type_ids = get_inputs(
                                context, query, tokenizer, title, max_len, [], window_head_entity)
                            self.t2_qas.append({"txt_ids": txt_ids, "context_mask": context_mask,
                                                "token_type_ids": token_type_ids})
                            self.t2_ids.append(
                                (passage_id, window_id, head_entity[:-1], rel, end_ent_type))
                            ofs = len(title) + \
                                len(tokenizer.tokenize(query)) + 3
                            self.query_offset2.append(ofs)

    def __len__(self):
        return len(self.t2_qas)

    def __getitem__(self, i):
        return self.t2_qas[i]


def load_data(dataset_tag, file_path, batch_size, max_len, pretrained_model_path, dist=False, shuffle=False, threshold=5):
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_path)
    dataset = MyDataset(dataset_tag, file_path, tokenizer,
                        max_len, threshold)
    sampler = DistributedSampler(dataset) if dist else None
    dataloader = DataLoader(dataset, batch_size, sampler=sampler, shuffle=shuffle if not sampler else False,
                            collate_fn=collate_fn)
    return dataloader


def reload_data(old_dataloader, batch_size, max_len, threshold, local_rank, shuffle=True):
    dataset = old_dataloader.dataset
    old_max_len, old_threshold = dataset.max_len, dataset.threshold
    if not(old_max_len == max_len and old_threshold == threshold):
        dataset.max_len = max_len
        dataset.threshold = threshold
        dataset.init_data()
    sampler = DistributedSampler(
        dataset, rank=local_rank) if local_rank != -1 else None
    dataloader = DataLoader(
        dataset, batch_size, sampler=sampler, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader


def load_t1_data(dataset_tag, test_path, pretrained_model_path, window_size, overlap, batch_size=10, max_len=512):
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_path)
    t1_dataset = T1Dataset(dataset_tag, test_path,
                           tokenizer, window_size, overlap, max_len)
    dataloader = DataLoader(t1_dataset, batch_size, collate_fn=collate_fn1)
    return dataloader


def load_t2_data(t1_dataset, t1_predict, batch_size=10, threshold=5):
    t2_dataset = T2Dataset(t1_dataset, t1_predict, threshold)
    dataloader = DataLoader(t2_dataset, batch_size, collate_fn=collate_fn1)
    return dataloader
