import pandas as pd 
import numpy as np 
import json, time 
from tqdm import tqdm 
from sklearn.metrics import accuracy_score, classification_report
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertModel, BertConfig, BertTokenizer, AdamW, get_cosine_schedule_with_warmup
import warnings
warnings.filterwarnings('ignore')
from model import RNN
from config import config
from kappa import quadratic_weighted_kappa
bert_path = "/Users/wangche/Desktop/Summary/Transformer/chinese_rbt3_pytorch"   
tokenizer = BertTokenizer.from_pretrained(bert_path)  
input_ids, input_masks, input_types,  = [], [], [] 
labels = []        
maxlen = config.max_essay_len
t=0
with open("./tes.csv", encoding='utf-8') as f:
    for i, line in tqdm(enumerate(f)): 
        if t==0:
            t=1
            continue
        title, y = line.strip().split(',')
        y=float(y)
        encode_dict = tokenizer.encode_plus(text=title, max_length=maxlen, 
                                            padding='max_length', truncation=True)
        input_ids.append(encode_dict['input_ids'])
        input_types.append(encode_dict['token_type_ids'])
        input_masks.append(encode_dict['attention_mask'])
        labels.append(y/100)
input_ids, input_types, input_masks = np.array(input_ids), np.array(input_types), np.array(input_masks)
labels = np.array(labels)
idxes = np.arange(input_ids.shape[0])
np.random.seed(2022)
np.random.shuffle(idxes)
print(idxes)
input_ids_train, input_ids_valid, input_ids_test = input_ids[idxes[:8899]], input_ids[idxes[8899:10011]], input_ids[idxes[10011:]]
input_masks_train, input_masks_valid, input_masks_test = input_masks[idxes[:8899]], input_masks[idxes[8899:10011]], input_masks[idxes[10011:]] 
input_types_train, input_types_valid, input_types_test = input_types[idxes[:8899]], input_types[idxes[8899:10011]], input_types[idxes[10011:]]
y_train, y_valid, y_test = labels[idxes[:8899]], labels[idxes[8899:10011]], labels[idxes[10011:]]
BATCH_SIZE = 64
print(y_test)
train_data = TensorDataset(torch.LongTensor(input_ids_train), 
                           torch.LongTensor(input_masks_train), 
                           torch.LongTensor(input_types_train), 
                           torch.FloatTensor(y_train))
train_sampler = RandomSampler(train_data)  
train_loader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)
valid_data = TensorDataset(torch.LongTensor(input_ids_valid), 
                          torch.LongTensor(input_masks_valid),
                          torch.LongTensor(input_types_valid), 
                          torch.FloatTensor(y_valid))
valid_sampler = SequentialSampler(valid_data)
valid_loader = DataLoader(valid_data, sampler=valid_sampler, batch_size=BATCH_SIZE)

test_data = TensorDataset(torch.LongTensor(input_ids_test), 
                          torch.LongTensor(input_masks_test),
                          torch.LongTensor(input_types_test),
                          torch.FloatTensor(y_test))
test_sampler = SequentialSampler(test_data)
test_loader = DataLoader(test_data, sampler=test_sampler, batch_size=BATCH_SIZE)



class Bert_Model(nn.Module):
    def __init__(self, bert_path, classes=20):
        super(Bert_Model, self).__init__()
        self.config = BertConfig.from_pretrained(bert_path)
        self.bert = BertModel.from_pretrained(bert_path)
        self.fc = nn.Linear(self.config.hidden_size, classes)
        
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        out_pool = outputs[1]
        logit = self.fc(out_pool)
        return logit
def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return 'Total parameters: {}, Trainable parameters: {}'.format(total_num, trainable_num)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 25
torch.manual_seed(0)
model = RNN(config)
print(get_parameter_number(model))
optimizer = AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=len(train_loader),
                                            num_training_steps=EPOCHS*len(train_loader))
def evaluate(model, data_loader, device):
    model.eval()
    val_true, val_pred = [], []
    with torch.no_grad():
        for idx, (ids, att, tpe, y) in (enumerate(data_loader)):
            y_pred = model(ids.to(device))
            y_pred*=20
            y_pred=torch.floor(y_pred)
            y_pred=y_pred.long()
            y_pred=[int(i[0]) for i in y_pred]
            val_pred.extend(y_pred)
            y=[int(i*20) for i in y]
            val_true.extend(y)
    print(val_true)
    print(val_pred)
    return quadratic_weighted_kappa(val_true, val_pred,20)
def predict(model, data_loader, device):
    model.eval()
    val_pred = []
    with torch.no_grad():
        for idx, (ids, att, tpe) in tqdm(enumerate(data_loader)):
            y_pred = model(ids.to(device))
            y_pred*=20
            y_pred=torch.floor(y_pred)
            y_pred=y_pred.long()
            y_pred=[int(i[0]) for i in y_pred]
            val_pred.extend(y_pred)
    return val_pred


def train_and_eval(model, train_loader, valid_loader,
                   optimizer,scheduler,device, epoch):
    best_acc = 0.0

    torch.manual_seed(0)
    criterion = nn.MSELoss()
    for i in range(epoch):
        start = time.time()
        model.train()
        print("========= Running Training Epoch {} =========".format(i+1))
        train_loss_sum = 0.0
        for idx, (ids, att, tpe, y) in enumerate(train_loader):
            ids, att, tpe, y = ids.to(device), att.to(device), tpe.to(device), y.to(device)  
            y_pred = model(ids)
            y_pred=torch.squeeze(y_pred)
            loss = criterion(y_pred, y)
            loss.requires_grad_(True)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_loss_sum += loss.item()
            if (idx + 1) % (len(train_loader)//5) == 0:
                print("Epoch {:04d} | Step {:04d}/{:04d} | Loss {:.4f} | Time {:.4f}".format(
                          i+1, idx+1, len(train_loader), train_loss_sum/(idx+1), time.time() - start))
        model.eval()
        acc = evaluate(model, valid_loader, device)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "best_bert_model.pth") 
        
        print("current kappa is {:.4f}, best kappa is {:.4f}".format(acc, best_acc))
        print("time costed = {}s \n".format(round(time.time() - start, 5)))
train_and_eval(model, train_loader, valid_loader, optimizer,scheduler,DEVICE, EPOCHS)
model.load_state_dict(torch.load("best_bert_model.pth"))
print("\n Test Kappa = {} \n".format(evaluate(model,test_loader,DEVICE)))
