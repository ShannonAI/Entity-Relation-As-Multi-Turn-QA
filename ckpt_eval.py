import argparse
import pickle
import os
import torch
from evaluation import test_evaluation,full_dev_evaluation
from model import MyModel
from dataloader import load_t1_data,load_data

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--window_size",type=int,default=300)
    parser.add_argument("--overlap",type=int,default=45)
    parser.add_argument("--checkpoint_path",required=True,default="./checkpoints/model_create_date/checkpoint_i.cpt")
    parser.add_argument("--test_path")
    parser.add_argument("--test_batch",default=10)
    parser.add_argument('--max_len',default=512)
    parser.add_argument("--gold_t1",action="store_true",help="在gold头实体上进行预测")
    parser.add_argument("--threshold",type=int,default=-1)
    parser.add_argument("--pretrained_model_path",default="")
    parser.add_argument("--amp",action='store_true')
    args = parser.parse_args()
    model_dir,file = os.path.split(args.checkpoint_path)
    config = pickle.load(open(os.path.join(model_dir,'args'),'rb'))
    checkpoint = torch.load(os.path.join(model_dir,file),map_location=torch.device("cpu"))
    model_state_dict = checkpoint['model_state_dict']
    #替换config的默认值
    config.pretrained_model_path = args.pretrained_model_path if args.pretrained_model_path else config.pretrained_model_path
    config.threshold = args.threshold if args.threshold==-1 else config.threshold
    mymodel = MyModel(config)
    mymodel.load_state_dict(model_state_dict,strict=False)
    device = torch.device("cuda") if  torch.cuda.is_available() else torch.device("cpu")
    mymodel.to(device)
    
    test_dataloader = load_t1_data(config.dataset_tag,args.test_path,config.pretrained_model_path,args.window_size,args.overlap,args.test_batch,args.max_len)
    (p1,r1,f1),(p2,r2,f2) = test_evaluation(mymodel,test_dataloader,config.threshold,args.gold_t1,args.amp)
    print("Turn 1: precision:{:.4f} recall:{:.4f} f1:{:.4f}".format(p1,r1,f1))
    print("Turn 2: precision:{:.4f} recall:{:.4f} f1:{:.4f}".format(p2,r2,f2))