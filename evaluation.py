import torch
from tqdm import tqdm

from dataloader import tag_idxs, load_t2_data


def get_score(gold_set, predict_set):
    TP = len(set.intersection(gold_set, predict_set))
    print("#TP:", TP, "#Gold:", len(gold_set), "#Predict:", len(predict_set))
    precision = TP/(len(predict_set)+1e-6)
    recall = TP/(len(gold_set)+1e-6)
    f1 = 2*precision*recall/(precision+recall+1e-6)
    return precision, recall, f1


def test_evaluation(model, t1_dataloader, threshold, amp=False):
    if hasattr(model, 'module'):
        model = model.module
    model.eval()
    t1_predict = []
    t2_predict = []
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    # turn 1
    with (torch.no_grad() if not amp else torch.cuda.amp.autocast()):
        for i, batch in enumerate(tqdm(t1_dataloader, desc="t1 predict")):
            txt_ids, attention_mask, token_type_ids, context_mask = batch['txt_ids'], batch[
                'attention_mask'], batch['token_type_ids'], batch['context_mask']
            tag_idxs = model(txt_ids.to(device), attention_mask.to(
                device), token_type_ids.to(device))
            predict_spans = tag_decode(tag_idxs, context_mask)
            t1_predict.extend(predict_spans)
    # turn 2
    t2_dataloader = load_t2_data(
        t1_dataloader.dataset, t1_predict, 10, threshold)
    with (torch.no_grad() if not amp else torch.cuda.amp.autocast()):
        for i, batch in enumerate(tqdm(t2_dataloader, desc="t2 predict")):
            txt_ids, attention_mask, token_type_ids, context_mask = batch['txt_ids'], batch[
                'attention_mask'], batch['token_type_ids'], batch['context_mask']
            tag_idxs = model(txt_ids.to(device), attention_mask.to(
                device), token_type_ids.to(device))
            predict_spans = tag_decode(tag_idxs, context_mask)
            t2_predict.extend(predict_spans)
    # get basic information
    t1_ids = t1_dataloader.dataset.t1_ids
    t2_ids = t2_dataloader.dataset.t2_ids
    window_offset_base = t1_dataloader.dataset.window_offset_base
    query_offset1 = t1_dataloader.dataset.query_offset1
    query_offset2 = t2_dataloader.dataset.query_offset2
    t1_gold = t1_dataloader.dataset.t1_gold
    t2_gold = t2_dataloader.dataset.t2_gold
    p1, r1, f1 = eval_t1(t1_predict, t1_gold, t1_ids,
                        query_offset1, window_offset_base)
    p2, r2, f2 = eval_t2(t2_predict, t2_gold, t2_ids,
                         query_offset2, window_offset_base)
    return (p1, r1, f1), (p2, r2, f2)


def eval_t1(predict, gold, ids, query_offset, window_offset_base):
    """
    Args:
        predict: [(s1,e1),(s2,e2),(s3,e3),...]
        gold: (passage_id,(entity_type,start_idx,end_idx,entity_str))
        ids: (passage_id, window_id,entity_type)
        query_offset: offset of [CLS]+title+[SEP]+query+[SEP]
        window_offset_base: value of window_size-overlap的值
    """
    predict1 = []
    for i, (_id, pre) in enumerate(zip(ids, predict)):
        passage_id, window_id, entity_type = _id
        window_offset = window_offset_base*window_id
        for start, end in pre:
            start1, end1 = start - query_offset[i]+window_offset, \
                            end - query_offset[i]+window_offset
            new = (passage_id, (entity_type, start1, end1))
            predict1.append(new)
    return get_score(set(gold), set(predict1))


def eval_t2(predict, gold, ids, query_offset, window_offset_base):
    """
    Args:
        predict: [(s1,e1),(s2,e2),(s3,e3),...]
        gold:  (passage_id,(head_entity,relation_type,end_entity))
        ids: (passage_id,window_id,head_entity,relation_type,end_entity_type)
        query_offset: offset of [CLS]+title+[SEP]+query+[SEP]
        window_offset_base: value of window_size-overlap
    """
    predict1 = []
    for i, (_id, pre) in enumerate(zip(ids, predict)):
        passage_id, window_id, head_entity, relation_type, end_entity_type = _id
        window_offset = window_offset_base*window_id
        head_start = head_entity[1]
        for start, end in pre:
            #since we added two special tokens around the head entity for identification, there is a correction of 1.
            if head_start+query_offset[i]-window_offset+1 < start:
                start1, end1 = start - \
                    query_offset[i]+window_offset-2, end - \
                    query_offset[i]+window_offset-2
            else:
                start1, end1 = start - \
                    query_offset[i]+window_offset, end - \
                    query_offset[i]+window_offset
            new = (passage_id, (head_entity, relation_type,
                                (end_entity_type, start1, end1)))
            predict1.append(new)
    return get_score(set(gold), set(predict1))


def tag_decode(tags, context_mask=None):
    spans = [[]]*tags.shape[0]
    tags = tags.tolist()
    if not context_mask is None:
        context_mask = context_mask.tolist()
    has_answer = []
    start_idxs = []
    end_idxs = []
    for i, t in enumerate(tags):
        if t[0] != tag_idxs['S']:
            has_answer.append(i)
            if context_mask is None:
                mask = [1 if i != -1 else 0 for i in t]
            else:
                mask = context_mask[i]
            s = mask.index(1, 1)
            e = mask.index(0, s)
            start_idxs.append(s)
            end_idxs.append(e)
    for i, s, e in zip(has_answer, start_idxs, end_idxs):
        span = []
        j = s
        while j < e:
            if tags[i][j] == tag_idxs['S']:
                span.append([j, j+1])
                j += 1
            elif tags[i][j] == tag_idxs['B'] and j < e-1:
                for k in range(j+1, e):
                    if tags[i][k] in [tag_idxs['B'], tag_idxs['S']]:
                        j = k
                        break
                    elif tags[i][k] == tag_idxs["E"]:
                        span.append([j, k+1])
                        j = k+1
                        break
                    elif k == e-1:
                        j = k+1
            else:
                j += 1
        spans[i] = span
    return spans
