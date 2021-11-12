import csv

import torch
from torch.utils.data import DataLoader
import numpy as np
from TitanicDataset import TitanicDataset
from NeuralNet import NeuralNet
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os
tr_path='train.csv'
tt_path='test.csv'

def get_device():
    ''' Get device (if GPU is available, use GPU) '''
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def plot_learning_curve(loss_record,title=''):
    total_steps=len(loss_record['train'])
    x_1=range(total_steps)
    figure(figsize=(6,4))
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.plot(x_1,loss_record['train'],c='tab:red',label='train')
    plt.xlabel('epoch')
    plt.ylabel('BCE Loss')
    plt.title('Learning curve of {}'.format(title))
    plt.legend()
    plt.show()

def pre_dataloader(path,mode,batch_size,num_worker):
    dataset=TitanicDataset(path,mode)
    dataloader=DataLoader(dataset=dataset,batch_size=batch_size,shuffle=(mode=="train"),\
                          num_workers=num_worker,pin_memory=True)

    return dataloader

def train(tr_set,model,config,device):
    n_epochs=config['n_epochs']
    optimizer=getattr(torch.optim,config['optimizer'])(
        model.parameters(),**config['optim_hparas'])

    loss_record={'train':[]}
    for epoch in range(n_epochs):
        for i,data in enumerate(tr_set,0):
            input,label=data
            input, label = input.to(device), label.to(device)
            y_pred=model(input)
            loss=model.cal_loss(y_pred,label)

            print(f'epoch:{epoch},{i},loss:{loss.item()}')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("**************************************************")
        loss_record['train'].append(loss.detach().cpu().item())

    torch.save(model.state_dict(),config['save_path'])
    return loss_record

def test(tt_set,model,device):
    model.eval()
    preds=[]
    for x in tt_set:
        x=x.to(device)
        with torch.no_grad():
            pred=model(x)
            preds.append(pred.detach().cpu())
    preds=torch.cat(preds,dim=0).numpy()
    return preds

def save_pred(preds,file):
    print('Saving results to {}'.format(file))
    with open(file,'w') as fp:
        writer=csv.writer(fp)
        writer.writerow(['id','tested_positive'])
        for i,p in enumerate(preds):
            p_=1 if p>=0.5 else 0
            writer.writerow([i,p_])


config={
    'n_epochs':100,
    'batch_size':32,
    'optimizer':'SGD',
    'optim_hparas':{
        'lr':0.01,
        'momentum':0.8
    },
    'save_path':'Mymodels/model.pth'
}

if __name__=="__main__":
    device=get_device()
    os.makedirs('Mymodels',exist_ok=True)
    tr_set = pre_dataloader(tr_path, 'train', config['batch_size'],0)
    tt_set = pre_dataloader(tt_path, 'test', config['batch_size'], 0)
    model = NeuralNet(3).to(device)
    model_loss_record=train(tr_set,model,config,device)
    plot_learning_curve(model_loss_record,title='deep model')
    #preds = test(tt_set, model, device)
    #save_pred(preds, 'pred.csv')
