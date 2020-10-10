import os
import time
import random
import argparse

import pickle
from transformers.optimization import get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast,GradScaler
from tqdm import trange,tqdm
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from model import MyModel
from evaluation import test_evaluation,full_dev_evaluation
from dataloader import load_data,load_t1_data,reload_data


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_tag",default='ACE2005',choices=['ace2005','ace2004'])
    parser.add_argument("--train_path",help="json数据的路径，或者dataloader的路径",default="/home/wangnan/mtqa4kg/data/cleaned_data/ACE2005/one_train.json")
    parser.add_argument("--train_batch",type=int,default=10)
    parser.add_argument("--dev_path",help="json数据的路径，或者dataloader的路径",default="/home/wangnan/mtqa4kg/data/cleaned_data/ACE2005/one_train.json")
    parser.add_argument("--dev_batch",type=int,default=10)
    parser.add_argument("--test_path",default="/home/wangnan/mtqa4kg/data/cleaned_data/ACE2005/one_test.json")
    parser.add_argument("--test_batch",type=int,default=10)
    parser.add_argument("--turn2_down_sample_ratio",default=-1,type=float,help="取值为0-1，代表每篇文章的")
    parser.add_argument("--dynamic_sample",action="store_true",help="是否每个epoch重新采样")
    parser.add_argument("--max_len",default=512,type=int,help="输入的最大长度")#这个参数和我们数据处理阶段的窗口大小有一定的关系
    parser.add_argument("--pretrained_model_path",default=r'/home/wangnan/pretrained_models/bert-base-uncased')
    parser.add_argument("--max_epochs",default=10,type=int)
    parser.add_argument("--warmup_ratio",type=float,default=-1)
    parser.add_argument("--lr",type=float,default=2e-5)
    parser.add_argument("--dropout_prob",type=float,default=0.1)
    parser.add_argument("--weight_decay",type=float,default=0.01)
    parser.add_argument("--theta",type=float,help="调节两个任务的权重",default=0.25)
    parser.add_argument("--window_size",type=int,default=100,help="滑窗的大小")
    parser.add_argument("--overlap",type=int,default=50,help="两个滑窗的重叠大小")
    parser.add_argument("--threshold",type=int,default=5,help="一个可能存在的关系在训练集至少应该出现的次数（应该大于等于数据预处理阶段的threshold）")
    parser.add_argument("--local_rank",type=int,default=-1,help="用于DistributedDataParallel")
    parser.add_argument("--max_grad_norm",type=float,default=1)
    parser.add_argument("--seed",type=int,default=209)
    parser.add_argument("--amp",action="store_true",help="是否开启混合精度")
    parser.add_argument("--not_save",action="store_true",help="是否保存模型")
    parser.add_argument("--reload",action="store_true",help="是否重新构造缓存的数据")
    parser.add_argument("--eval",action="store_true",help="是否在验证集上评估模型(目前统一当作NER任务评估)")
    parser.add_argument("--test_eval",action="store_true",help="在测试集上进行评估，包含第一轮NER和第二轮RE的评估")
    parser.add_argument("--tensorboard",action="store_true",help="是否开启tensorboard")
    args = parser.parse_args()
    return args


def reduce_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """DDP的时候各个进程的Loss的平均值"""
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    rt /= torch.distributed.get_world_size()#进程数
    return rt

def train(args,train_dataloader,dev_dataloader=None):
    model = MyModel(args)
    model.train()
    if args.amp:
        scaler = GradScaler()
    device = args.local_rank if args.local_rank!=-1 else (torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu'))
    if args.local_rank!=-1:
        torch.cuda.set_device(args.local_rank)
    model.to(device)
    if args.local_rank!=-1:
        model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[args.local_rank],output_device=args.local_rank,find_unused_parameters=True)
    no_decay = ['bias','LayerNorm.weight']
    optimizer_grouped_parameters = [
        {"params":[p for n,p in model.named_parameters() if not any(nd in n for nd in no_decay)],"weight_decay":args.weight_decay},
        {"params":[p for n,p in model.named_parameters() if any(nd in n for nd in no_decay)],"weight_decay":0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters,lr=args.lr)
    if args.warmup_ratio>0:
        num_training_steps = len(train_dataloader)*args.max_epochs
        warmup_steps = args.warmup_ratio*num_training_steps
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, num_training_steps)
    if args.local_rank<1:
        mid = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(time.time()))
        if args.tensorboard:
            log_dir = "./logs/{}/{}".format(args.dataset_tag,mid)
            writer = SummaryWriter(log_dir)
    for epoch in range(args.max_epochs):
        if args.local_rank!=-1:
            train_dataloader.sampler.set_epoch(epoch)
        if  args.dynamic_sample and  1>=args.turn2_down_sample_ratio>0:
            train_dataloader.dataset.set_epoch(epoch)
            train_dataloader.dataset.t2_down_sample()
        tqdm_train_dataloader = tqdm(train_dataloader,desc="epoch:%d"%epoch,ncols=150)
        for i,batch in enumerate(tqdm_train_dataloader):
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            txt_ids, attention_mask, token_type_ids, context_mask, turn_mask, tags=batch['txt_ids'],batch['attention_mask'], batch['token_type_ids'],\
                                                                          batch['context_mask'],batch['turn_mask'],batch['tags']
            txt_ids, attention_mask, token_type_ids, context_mask, turn_mask, tags = txt_ids.to(device),attention_mask.to(device),token_type_ids.to(device),\
                                                                               context_mask.to(device),turn_mask.to(device),tags.to(device)
            if args.amp:
                with autocast():
                    loss,(loss_t1,loss_t2) = model(txt_ids, attention_mask, token_type_ids, context_mask, turn_mask,tags)
                scaler.scale(loss).backward()
                if args.max_grad_norm>0:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss,(loss_t1,loss_t2) = model(txt_ids, attention_mask, token_type_ids, context_mask, turn_mask,tags)
                loss.backward()
                if args.max_grad_norm>0:
                    clip_grad_norm_(model.parameters(),args.max_grad_norm)
                optimizer.step()
            lr = optimizer.param_groups[0]['lr']
            named_parameters = [(n,p) for n,p in model.named_parameters() if not p.grad is None]
            grad_norm = torch.norm(torch.stack([torch.norm(p.grad) for n,p in named_parameters])).item()
            if args.warmup_ratio>0:
                scheduler.step()
            reduced_loss = reduce_tensor(loss.data) if args.local_rank!=-1 else loss.item()
            if args.local_rank<1 and args.tensorboard:
                writer.add_scalars("loss",{'overall':reduced_loss,'turn1':loss_t1,'turn2':loss_t2},epoch*len(train_dataloader)+i)
                writer.add_scalars("lr_grad",{"lr":lr,"grad_norm":grad_norm},epoch*len(train_dataloader)+i)
                writer.flush()
            postfix_str = "norm:{:.2f},lr:{:.1e},loss:{:.2e},t1:{:.2e},t2:{:.2e}".format(grad_norm,lr,reduced_loss,loss_t1,loss_t2)
            tqdm_train_dataloader.set_postfix_str(postfix_str)
        if args.local_rank in [-1,0] and not args.not_save:
            if hasattr(model,'module'):
                model_state_dict = model.module.state_dict()
            else:
                model_state_dict = model.state_dict()
            checkpoint = {"model_state_dict":model_state_dict}
            save_dir = './checkpoints/%s/%s/'%(args.dataset_tag,mid)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                pickle.dump(args,open(save_dir+'args','wb'))
            save_path = save_dir+"checkpoint_%d.cpt"%epoch
            torch.save(checkpoint,save_path)
            print("model saved at:",save_path)
        if args.eval and args.local_rank in [-1,0]:
            p,r,f = full_dev_evaluation(model,dev_dataloader,args.amp)
            print("precision:{:.4f} recall:{:.4f} f1:{:.4f}".format(p, r, f))
            if args.tensorboard:
                writer.add_scalars("score", {"precision": p, "recall": r, 'f1': f}, epoch)
                writer.flush()
            model.train()
        if args.test_eval and args.local_rank in [-1,0]:
            test_dataloader = load_t1_data(args.dataset_tag,args.test_path,args.pretrained_model_path,args.window_size,args.overlap,args.test_batch,args.max_len) #test_dataloader是第一轮问答的dataloder
            (p1,r1,f1),(p2,r2,f2) = test_evaluation(model,test_dataloader,args.threshold,False,args.amp)
            print("Turn 1: precision:{:.4f} recall:{:.4f} f1:{:.4f}".format(p1,r1,f1))
            print("Turn 2: precision:{:.4f} recall:{:.4f} f1:{:.4f}".format(p2,r2,f2))
            model.train()
        if args.local_rank!=-1:
            torch.distributed.barrier()
    if args.local_rank in [-1,0] and args.tensorboard:
        writer.close()

if __name__=="__main__":
    args = args_parser()
    set_seed(args.seed)
    print(args)
    if args.local_rank!=-1:
        torch.distributed.init_process_group(backend='nccl')
    if args.train_path.endswith(".json"):
        p = '{}_{}_{}'.format(args.dataset_tag, os.path.split(args.train_path)[-1].split('.')[0],os.path.split(args.pretrained_model_path)[-1])	
        p1 = os.path.join(os.path.split(args.train_path)[0],p)	
        if not os.path.exists(p1) or args.reload:	
            train_dataloader = load_data(args.dataset_tag ,args.train_path, args.train_batch, args.max_len, args.pretrained_model_path,	
                                         args.local_rank != -1, shuffle=True,down_sample_ratio=args.turn2_down_sample_ratio,threshold=args.threshold)	
            pickle.dump(train_dataloader,open(p1,'wb'))	
            print("training data saved at ",p1)	
        else:	
            print("reload training data from ",p1)	
            train_dataloader = pickle.load(open(p1, 'rb'))	
            train_dataloader = reload_data(train_dataloader,args.train_batch,args.max_len,args.turn2_down_sample_ratio,args.threshold,args.local_rank,True)	
            pickle.dump(train_dataloader,open(p1,'wb'))	
    if args.eval:	
        #dev对第二轮问答的评估也是在gold上进行的，不能完全反映联合关系抽取的情况
        if args.dev_path.endswith('.json'):	
            p = '{}_{}_{}'.format(args.dataset_tag,os.path.split(args.dev_path)[-1].split('.')[0],os.path.split(args.pretrained_model_path)[-1])	
            p1 = os.path.join(os.path.split(args.dev_path)[0], p)	
            if not os.path.exists(p1) or args.reload:	
                dev_dataloader = load_data(args.dataset_tag, args.dev_path,args.dev_batch,args.max_len,args.pretrained_model_path,False,False,-1,threshold=args.threshold)	
                pickle.dump(dev_dataloader,open(p1,'wb'))	
                print("evaluation data saved at ",p1)	
            else:	
                print("reload evaluation data from ",p1)	
                dev_dataloader = pickle.load(open(p1, "rb"))	
                dev_dataloader = reload_data(dev_dataloader,args.dev_batch,args.max_len,-1,args.threshold,-1,False)
                pickle.dump(dev_dataloader,open(p1,'wb'))	
    else:	
        dev_dataloader = None	
    train(args,train_dataloader,dev_dataloader)