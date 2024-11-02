import os
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader,Dataset
import ast

nt_int={'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10, 'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20, 'X': 21}


def seq2int(nt_sequence,target_length=1024):
    int_sequence=[]
    for nt in nt_sequence:
        nt=nt.upper()
        if nt in nt_int:
            int_sequence.append(nt_int[nt])
    int_sequence=np.asarray(int_sequence,dtype='int32')
    if len(int_sequence) > target_length:
        int_sequence = int_sequence[:target_length]
    if len(int_sequence) < target_length:
        int_sequence=np.pad(int_sequence,(0,target_length-len(int_sequence)),constant_values=0)
    return int_sequence

class MyDataset(Dataset):
    def __init__(self, file):
        self.embedding, self.seq, self.label = self.read_file(file)

    def read_file(self,file_path):
        embedding = []
        labels = []
        seq = []
        df_all = pd.read_csv(file_path)
        df_all['SeqVec'] = df_all['SeqVec'].apply(lambda x: np.array(ast.literal_eval(x)))
        for i in range(len(df_all)):
            embedding.append(df_all['SeqVec'][i])
            labels.append(df_all['Label'][i])
            # tdi
            seq.append(seq2int(df_all['Seq'][i]))
        return embedding, seq, labels

    def __len__(self):
        return len(self.embedding)

    def __getitem__(self, index):

        embedding=self.embedding[index]
        label=int(self.label[index])
        seq = self.seq[index]
        return embedding, seq, label
