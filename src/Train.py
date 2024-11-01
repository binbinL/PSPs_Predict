import os
import torch
import wandb
import time
from tqdm import tqdm
import pandas as pd
import numpy as np
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
import ast
from MyNet import MyModel,MyModel2,MyModel3,MyModel4 # type: ignore
from LrScheduler import WarmupLR # type: ignore
from Func import analysis # type: ignore
from config import train_root,val_root, batch_size,n_epoch,lr,dropout,save_root,d_model,n_class,vocab_size,nlayers,nhead,dim_feedforward,d_embedding # type: ignore
from MyDataSet import MyDataset # type: ignore



def evaluate(model, loader):
    model.eval()

    loss_list = []
    labels_list = []
    preds_list = []
    
    with torch.no_grad():
        for data, seq, labels in loader: # 生成一个 batch 的数据和标注
            data = data.to(device)
            labels = labels.to(device)
            seq = seq.to(device)
            
            outputs = model(data,seq)

            loss = criterion(outputs, labels) 

            probabilities = nn.functional.softmax(outputs, dim=1)
            preds = probabilities[:, 1].detach().cpu().numpy()
            
            loss = loss.detach().cpu().numpy()
            outputs = outputs.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()

            loss_list.append(loss)
            labels_list.extend(labels)
            preds_list.extend(preds)
        
    # 计算分类评估指标
    epoch_loss_avg  = np.mean(loss_list)

    return epoch_loss_avg,labels_list,preds_list


def train():

    TrainData = MyDataset(train_root)
    TestData = MyDataset(val_root)

    TrainDataLoader = DataLoader(TrainData, batch_size=batch_size,shuffle=True)
    TestDataLoader = DataLoader(TestData, batch_size=batch_size,shuffle=True)

    best_roc_auc_score = 0.0
    # 训练日志-训练集
    df_train_log = pd.DataFrame()
    # 训练日志-测试集
    df_test_log = pd.DataFrame()
    # lr
    df_lr_log = pd.DataFrame()

    wandb.init(project='LLPS', name=time.strftime('%m%d%H%M%S'))
    wandb.watch(model, log="gradients", log_freq=1000, log_graph=False)
    for epoch in range(1, n_epoch+1):
        print(f'Epoch {epoch}/{n_epoch}')
        model.train()
        loss_list = []
        
        for data, seq, labels in tqdm(TrainDataLoader):
            data = data.to(device)
            seq = seq.to(device)
            labels = labels.to(device)
            outputs = model(data,seq)

            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()#梯度清零
            loss.backward()#反向传播
            optimizer.step()#更新权重

            loss = loss.detach().cpu().numpy()
            loss_list.append(loss)

            log_lr = {}
            log_lr['epoch'] = epoch
            log_lr['lr'] = optimizer.param_groups[0]['lr']
            df_lr_log = pd.concat([df_lr_log,pd.DataFrame([log_lr])],ignore_index=True)
            wandb.log(log_lr)  
            #学习率调整

            lr_scheduler.step()

        
        # "========== Evaluate Train set =========="
        epoch_loss_train_avg ,train_label,train_pred = evaluate(model, TrainDataLoader)
        result_train = analysis(train_label, train_pred)

        log_train = {}
        log_train['epoch'] = epoch
        log_train['train_loss'] = np.sqrt(epoch_loss_train_avg)
        log_train['train_accuracy'] = result_train['binary_acc']
        log_train['train_precision'] = result_train['precision']
        log_train['train_recall'] = result_train['recall']
        log_train['train_f1-score'] = result_train['f1']
        log_train['train_roauc_score'] = result_train['roauc']
        log_train['train_prauc_score'] = result_train['prauc']
        log_train['train_mcc'] = result_train['mcc']
        log_train['train_sensitivity'] = result_train['sensitivity']
        log_train['train_specificity'] = result_train['specificity']
        df_train_log = pd.concat([df_train_log,pd.DataFrame([log_train])],ignore_index=True)
        wandb.log(log_train)

        # "========== Evaluate Valid set =========="
        epoch_loss_valid_avg, valid_label, valid_pred = evaluate(model, TestDataLoader)
        result_valid = analysis(valid_label, valid_pred)

        log_test = {}
        log_test['epoch'] = epoch
        log_test['valid_loss'] = np.sqrt(epoch_loss_valid_avg)
        log_test['valid_accuracy'] = result_valid['binary_acc']
        log_test['valid_precision'] = result_valid['precision']
        log_test['valid_recall'] = result_valid['recall']
        log_test['valid_f1-score'] = result_valid['f1']
        log_test['valid_roauc_score'] = result_valid['roauc']
        log_test['valid_prauc_score'] = result_valid['prauc']
        log_test['valid_mcc'] = result_valid['mcc']
        log_test['valid_sensitivity'] = result_valid['sensitivity']
        log_test['valid_specificity'] = result_valid['specificity']
        df_test_log = pd.concat([df_test_log,pd.DataFrame([log_test])],ignore_index=True)
        wandb.log(log_test)

        # 保存最新的最佳模型文件
        if log_test['valid_roauc_score'] > best_roc_auc_score: 
            # 删除旧的最佳模型文件(如有)
            old_best_checkpoint_path = save_root+'/best-{:.3f}.pth'.format(best_roc_auc_score)
            if os.path.exists(old_best_checkpoint_path):
                os.remove(old_best_checkpoint_path)
            # 保存新的最佳模型文件
            best_roc_auc_score = log_test['valid_roauc_score']
            new_best_checkpoint_path = save_root+'/best-{:.3f}.pth'.format(log_test['valid_roauc_score'])

            if not os.path.exists(save_root):
                os.makedirs(save_root)

            torch.save(model.state_dict(), new_best_checkpoint_path)
            print('保存新的最佳模型', save_root+'/best-{:.3f}.pth'.format(best_roc_auc_score))

    #保存日志数据
    save_root_dir = save_root+'/best-{:.3f}'.format(best_roc_auc_score)
    if not os.path.exists(save_root_dir):
        os.makedirs(save_root_dir)


    df_train_log.to_csv(save_root_dir+'/train_log.csv', index=False)
    df_test_log.to_csv(save_root_dir+'/val_log.csv', index=False)
    df_lr_log.to_csv(save_root_dir+'/lr_log.csv', index=False)


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
#d_model,nlayers,nhead,dropout,dim_feedforward,n_class
# model = MyModel(d_model,4,4,dropout,1024,n_class)

# d_model, dropout, n_class, vocab_size,nlayers,nhead,dim_feedforward
model = MyModel4(d_embedding,d_model,dropout,n_class,vocab_size,nlayers,nhead,dim_feedforward)
model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(),lr=lr, weight_decay=0.1)
# optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
# lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
# lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer = optimizer,T_max = n_epoch*31/7,eta_min=1e-6,verbose=True)
# ['triangular','triangular2','exp_range']
# lr_scheduler = optim.lr_scheduler.CyclicLR(optimizer=optimizer,base_lr=1e-5, max_lr=1e-1, 
#                                            step_size_up=200, mode='exp_range', gamma=0.9,cycle_momentum=False)
# ['cos','linear']
# lr_scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=1e-1, epochs=n_epoch,steps_per_epoch=31,anneal_strategy='cos')

lr_scheduler = WarmupLR(optimizer=optimizer, num_warm=100)

if __name__ == '__main__':
    train()
