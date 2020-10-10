import torch
import torch.nn as nn
from transformers import BertModel


class MyModel(nn.Module):
    def __init__(self,config):
        super(MyModel,self).__init__()
        self.config=config
        self.bert = BertModel.from_pretrained(config.pretrained_model_path)
        self.tag_linear = nn.Linear(self.bert.config.hidden_size,5)
        self.dropout = nn.Dropout(config.dropout_prob)
        self.loss_func = nn.CrossEntropyLoss()
        self.theta = config.theta

    def forward(self,input,attention_mask,token_type_ids,context_mask=None,turn_mask=None,target_tags=None):
        """
        Args:
            input: （batch,seq_len），batch里面可能有第一轮的问答，也可能有第二轮的问答
            attention_mask: (batch,seq_len)
            token_type_ids: (batch,seq_len)
            context_mask: (batch,seq_len)，context用来确认拥有标注的token，注意为了处理无答案的情况[CLS]也属于context
            target_tags: (batch,seq_len)
            turn_mask: (batch,) turn_mask[i]=0代表第一轮，turn_mask[i]=1代表第2轮
        """
        rep,_ = self.bert(input,attention_mask,token_type_ids)
        rep = self.dropout(rep)
        tag_logits = self.tag_linear(rep) #(batch,seq_len,5)
        if not target_tags is None:
            #训练的情形
            tag_logits_t1 = tag_logits[turn_mask==0]#(n1,seq_len,num_tag)
            target_tags_t1 = target_tags[turn_mask==0]#(n1,seq_len)
            context_mask_t1 = context_mask[turn_mask==0]#(n1,seq_len)

            tag_logits_t2 = tag_logits[turn_mask==1]#(n2,seq_len,num_tag)
            target_tags_t2 = target_tags[turn_mask==1]#(n2,seq_len)
            context_mask_t2 = context_mask[turn_mask==1]#(n2,seq_len)

            tag_logits_t1 = tag_logits_t1[context_mask_t1==1]#(N1,num_tag)
            target_tags_t1 = target_tags_t1[context_mask_t1==1]#(N1)

            tag_logits_t2 = tag_logits_t2[context_mask_t2==1]#(N2,num_tag)
            target_tags_t2 = target_tags_t2[context_mask_t2==1]#(N2)
            
            #batch里没有t1或t2时，不特殊处理的话t1或t2的loss会变为nan
            loss_t1 = self.loss_func(tag_logits_t1,target_tags_t1) if len(target_tags_t1)!=0 else torch.tensor(0).type_as(input)
            loss_t2 = self.loss_func(tag_logits_t2,target_tags_t2) if len(target_tags_t2)!=0 else torch.tensor(0).type_as(input)
            loss = self.theta*loss_t1+(1-self.theta)*loss_t2
            return loss,(loss_t1.item(),loss_t2.item())#后面一项主要用于训练的时候进行记录子任务的损失
        else:
            #预测的情形
            tag_idxs = torch.argmax(tag_logits,dim=-1).squeeze(-1)#(batch,seq_len)
            return tag_idxs