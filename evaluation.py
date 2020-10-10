import torch
from tqdm import tqdm
from dataloader import tag_idxs,load_t2_data



def get_score(gold_set,predict_set):
    """得到两个集合的precision,recall.f1"""
    #print("len gold",len(gold_set),"len predict",len(predict_set))
    TP = len(set.intersection(gold_set,predict_set))
    precision = TP/(len(predict_set)+1e-6)
    recall = TP/(len(gold_set)+1e-6)
    f1 = 2*precision*recall/(precision+recall+1e-6)
    return precision,recall,f1


def full_dev_evaluation(model,dataloader,amp=False):
    """直接当作NER的评估，不考虑第二轮问答"""
    if hasattr(model,'module'):
        model = model.module
    model.eval()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    gold = []
    t1_gold = []
    t2_gold = []
    predict = []
    t1_predict = []
    t1_gold = []
    t2_predict = []
    t2_gold = []
    tqdm_dataloader = tqdm(dataloader,desc="dev eval")
    with (torch.no_grad() if not amp else torch.cuda.amp.autocast()):
        for i,batch in enumerate(tqdm_dataloader):
            txt_ids, attention_mask, token_type_ids, context_mask, turn_mask,tags=batch['txt_ids'],batch['attention_mask'],batch['token_type_ids'],\
                                                                           batch['context_mask'],batch['turn_mask'],batch['tags']
            tag_idxs = model(txt_ids.to(device), attention_mask.to(device), token_type_ids.to(device))
            predict_spans = tag_decode(tag_idxs,context_mask)
            gold_spans = tag_decode(tags)
            turn1_predict = [p for i,p in enumerate(predict_spans) if turn_mask[i]==0]
            turn1_gold = [g for i,g in enumerate(gold_spans) if turn_mask[i]==0]
            turn2_predict = [p for i,p in enumerate(predict_spans) if turn_mask[i]==1]
            turn2_gold = [g for i,g in enumerate(gold_spans) if turn_mask[i]==1]
            predict.append((i,predict_spans))
            gold.append((i,gold_spans))
            t1_predict.append((i,turn1_predict))
            t1_gold.append((i,turn1_gold))
            t2_predict.append((i,turn2_predict))
            t2_gold.append((i,turn2_gold))
    gold2 = set()
    predict2 = set()
    for g in gold:
        i,gold_spans = g
        for j,gs in enumerate(gold_spans):
            for gsi in gs:
                item = (i,j,gsi[0],gsi[1])
                gold2.add(item)
    for p in predict:
        i,pre_spans = p
        for j, ps in enumerate(pre_spans):
            for psi in ps:
                item = (i,j, psi[0],psi[1])
                predict2.add(item)
    precision,recall,f1 = get_score(gold2,predict2)
    print("overall p,r,f:",precision,recall,f1)
    t1_gold2 = set()
    t1_predict2 = set()
    for g in t1_gold:
        i,gold_spans = g
        for j,gs in enumerate(gold_spans):
            for gsi in gs:
                item = (i,j,gsi[0],gsi[1])
                t1_gold2.add(item)
    for p in t1_predict:
        i,pre_spans = p
        for j, ps in enumerate(pre_spans):
            for psi in ps:
                item = (i,j, psi[0],psi[1])
                t1_predict2.add(item)
    t1_precision,t1_recall,t1_f1 = get_score(t1_gold2,t1_predict2)
    print("turn1 p,r,f:",t1_precision,t1_recall,t1_f1)
    t2_gold2 = set()
    t2_predict2 = set()
    for g in t2_gold:
        i,gold_spans = g
        for j,gs in enumerate(gold_spans):
            for gsi in gs:
                item = (i,j,gsi[0],gsi[1])
                t2_gold2.add(item)
    for p in t2_predict:
        i,pre_spans = p
        for j, ps in enumerate(pre_spans):
            for psi in ps:
                item = (i,j, psi[0],psi[1])
                t2_predict2.add(item)
    t2_precision,t2_recall,t2_f1 = get_score(t2_gold2,t2_predict2)
    #验证集上的t2评估相当于在gold entity上进行t2的评估
    print("turn2 p,r,f:",t2_precision,t2_recall,t2_f1)
    return precision,recall,f1

def test_evaluation(model,t1_dataloader,threshold,gold_t1=False,amp=False):
    if hasattr(model,'module'):
        model = model.module
    model.eval()
    t1_predict = []
    t2_predict = []
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    #第一轮问答
    with (torch.no_grad() if not amp else torch.cuda.amp.autocast()):
        for i,batch in enumerate(tqdm(t1_dataloader,desc="t1 predict")):
            txt_ids,attention_mask,token_type_ids,context_mask = batch['txt_ids'],batch['attention_mask'],batch['token_type_ids'],batch['context_mask']
            tag_idxs = model(txt_ids.to(device),attention_mask.to(device),token_type_ids.to(device))
            predict_spans = tag_decode(tag_idxs,context_mask)
            t1_predict.extend(predict_spans)
    #进行第二轮问答
    t2_dataloader = load_t2_data(t1_dataloader.dataset,t1_predict,10,threshold,gold_t1)
    with (torch.no_grad() if not amp else torch.cuda.amp.autocast()):
        for i,batch in enumerate(tqdm(t2_dataloader,desc="t2 predict")):
            txt_ids,attention_mask,token_type_ids,context_mask = batch['txt_ids'],batch['attention_mask'],batch['token_type_ids'],batch['context_mask']
            tag_idxs = model(txt_ids.to(device),attention_mask.to(device),token_type_ids.to(device))
            predict_spans = tag_decode(tag_idxs,context_mask)
            t2_predict.extend(predict_spans)
    #获取一些需要需要的信息
    t1_ids = t1_dataloader.dataset.t1_ids
    t2_ids = t2_dataloader.dataset.t2_ids
    window_offset_base = t1_dataloader.dataset.window_offset_base
    query_offset1 = t1_dataloader.dataset.query_offset1
    query_offset2 = t2_dataloader.dataset.query_offset2
    t1_gold = t1_dataloader.dataset.t1_gold
    t2_gold = t2_dataloader.dataset.t2_gold
    #第一阶段的评估，即评估我们的ner的结果
    p1,r1,f1 = eval_t(t1_predict,t1_gold,t1_ids,query_offset1,window_offset_base,False)
    #第二阶段的评估，即评估我们的ner+re的综合结果
    p2,r2,f2 = eval_t2(t2_predict,t2_gold,t2_ids,query_offset2,window_offset_base)
    return (p1,r1,f1),(p2,r2,f2)


def eval_t(predict,gold,ids,query_offset,window_offset_base,turn2=False):
    """
    Args:
        predict: [(s1,e1),(s2,e2),(s3,e3),...]
        gold: (passage_id,(entity_type,start_idx,end_idx,entity_str)) or (passage_id,(head_entity,relation_type,end_entity))
        ids: (passage_id, window_id,entity_type) or ，其中head_entity中的实体索引是passage而非context中的
        query_offset: [CLS]+title+[SEP]+query+[SEP]对应的offset
        window_offset_base: window_size-overlap的值
        turn2: 是否为第二轮评估，默认为第一轮的评估
    """
    predict1 = []
    for i,(_id,pre) in enumerate(zip(ids,predict)):
        if not turn2:
            passage_id, window_id, entity_type = _id
        else:
            passage_id, window_id, head_entity, relation_type, end_entity_type = _id
        window_offset = window_offset_base*window_id
        for start,end in pre:
            start1,end1 = start-query_offset[i]+window_offset,end-query_offset[i]+window_offset
            if not turn2:
                new = (passage_id,(entity_type,start1,end1))
            else:
                new = (passage_id,(head_entity,relation_type,(end_entity_type,start1,end1)))
            predict1.append(new)
    return get_score(set(gold),set(predict1))


def eval_t2(predict,gold,ids,query_offset,window_offset_base):
    """
    Args:
        predict: [(s1,e1),(s2,e2),(s3,e3),...]
        gold:  (passage_id,(head_entity,relation_type,end_entity))
        ids: (passage_id,window_id,head_entity,relation_type,end_entity_type)，这个head_entity中的碎银是对应未滑窗的passage中的索引
        query_offset: [CLS]+title+[SEP]+query+[SEP]对应的offset
        window_offset_base: window_size-overlap的值
    """
    predict1 = []
    for i,(_id,pre) in enumerate(zip(ids,predict)):
        passage_id, window_id, head_entity, relation_type, end_entity_type = _id
        window_offset = window_offset_base*window_id
        head_start = head_entity[1]
        for start,end in pre:
            #在head entity右侧的由于我们对head entity周围添加了特殊字符用以标识，这里要修正一下
            if head_start+query_offset[i]-window_offset+1<start:
                start1,end1 = start-query_offset[i]+window_offset-2,end-query_offset[i]+window_offset-2
            else:
                start1,end1 = start-query_offset[i]+window_offset,end-query_offset[i]+window_offset
            new = (passage_id,(head_entity,relation_type,(end_entity_type,start1,end1)))
            predict1.append(new)
    return get_score(set(gold),set(predict1))


def tag_decode(tags,context_mask=None):
    spans = [[]]*tags.shape[0]
    tags = tags.tolist()
    if not context_mask is None:
        context_mask = context_mask.tolist()
    #确定有答案的样本，以及对应的起点
    has_answer = []
    start_idxs = []
    end_idxs = []
    for i,t in enumerate(tags):
        if t[0]!=tag_idxs['S']:
            has_answer.append(i)
            if context_mask is None:
              mask = [1 if i!=-1 else 0 for i in t]
            else:
              mask = context_mask[i]
            s = mask.index(1,1)
            e = mask.index(0,s)
            start_idxs.append(s)
            end_idxs.append(e)
    for i,s,e in zip(has_answer,start_idxs,end_idxs):
        span = []
        j=s
        while j<e:
            if tags[i][j]==tag_idxs['S']:
                span.append([j,j+1])
                j+=1
            elif tags[i][j]==tag_idxs['B'] and j<e-1:
                #不是语法严格的解码，只要在遇到下一个B和S之前找到E就行(前期预测的结果很可能是语法不正确的)
                for k in range(j+1,e):
                    if tags[i][k] in [tag_idxs['B'],tag_idxs['S']]:
                        j=k
                        break
                    elif tags[i][k]==tag_idxs["E"]:
                        span.append([j,k+1])
                        j=k+1
                        break
                    elif k==e-1:
                        #到末尾了，也没有找到的情况
                        j=k+1
            else:
                j+=1
        spans[i]=span
    return spans