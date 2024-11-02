import torch # type: ignore
from tqdm import tqdm # type: ignore
import pandas as pd # type: ignore
from torch import nn # type: ignore
from torch.utils.data import DataLoader # type: ignore
from MyNet import MyModel4 # type: ignore
from config import dropout,d_model,n_class,vocab_size,nlayers,nhead,dim_feedforward,d_embedding # type: ignore
from MyDataSet import MyDataset # type: ignore


def evaluate(model, loader):
    model.eval()
    preds_list = []
    with torch.no_grad():
        for data, seq, labels in tqdm(loader): 
            data = data.to(device)
            seq = seq.to(device)
            outputs = model(data,seq)

            probabilities = nn.functional.softmax(outputs, dim=1)
            preds = probabilities[:, 1].detach().cpu().numpy()
            preds_list.extend(preds)
        
    return preds_list


def predict(src,tgt):

    Data = MyDataset(src)

    pDataLoader = DataLoader(Data, batch_size=32)

    df = pd.DataFrame()
    preds_list = evaluate(model, pDataLoader)

    df['preds'] = preds_list
    df.to_csv(tgt, index=False)



device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
model = MyModel4(d_embedding,d_model,dropout,n_class,vocab_size,nlayers,nhead,dim_feedforward)
state_dict = torch.load('/data/lwb/WorkSpace/PSPs/Net/res/best-0.901.pth')
model.load_state_dict(state_dict)

model = model.to(device)

if __name__ == '__main__':
    root = '/data/lwb/WorkSpace/PSPs/Net/data/Human_embedding.csv'
    tgt = '/data/lwb/WorkSpace/PSPs/Net/res/best-0.901/predict_all.csv'
    predict(root,tgt)
