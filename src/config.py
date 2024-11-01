"""config"""
train_root = '/data/lwb/WorkSpace/PSPs/Net/data/train_tdi.csv'
val_root = '/data/lwb/WorkSpace/PSPs/Net/data/test_tdi.csv'
save_root = '/data/lwb/WorkSpace/PSPs/Net/res'

batch_size = 32    # 一次训练所选取的样本数
lr = 1e-2             # 学习率
n_epoch = 100          # 训练次数
dropout = 0.6   
d_embedding = 1024 # ProtT5 dim
d_model = 256   # 词向量维度
n_class = 2    # 输出
vocab_size = 21   # 词典大小
nlayers = 4   # transformer encoder layer
nhead = 2    # transformer encoder head
dim_feedforward  =1024 # transformer encoder feedforward
