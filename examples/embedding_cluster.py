import sys

sys.path.insert(0, "../")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from OutlierDetection.Embedding import EntityEmbedding
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import pprint
import time

path = "~/dataset/train_withapi_01082020.pkl"
df = pd.read_pickle(path)

day_dict = {
    'MON' : 0,
    'TUE' : 1,
    'WED' : 2,
    'THU' : 3,
    'FRI' : 4,
    'SAT' : 5,
    'SUN' : 6
}

def pre_process(df, inplace=True):
    if not inplace:
        df = df.copy()
    df['APIKEY'] = df['APIKEY'].apply(lambda x: x.replace("APIKEY", "")).astype(int)
    df['TIMEBIN'] = df['TIMEBIN'].apply(lambda x: x.replace("BIN", "")).astype(int)
    df['API'] = df['API'].apply(lambda x: x.replace("API", "")).astype(int)
    df['DAY'] = df['DAY'].apply(lambda x : day_dict[x]).astype(int)
    df.drop(columns=['ANAMOLYDISTNUM', 'LABEL', 'NUMFAILURES'], inplace=True)

    df = df.dropna()
    return df

pre_process(df)

cat = ['APIKEY', 'API', 'DAY', 'TIMEBIN']
target = ['NUMREQUESTS']


print(df.head())
print("------------------------------------------------- \nNum Uniques: ")
for c in cat:
    print("{} : {}".format(c, df[c].nunique()))

# splitting the dataset into train and validation
X_train, X_test, Y_train, Y_test = train_test_split(df[cat].values, np.log1p(df[target].values), test_size=0.2)

# print(X_train.shape, X_test.shape)
# print(Y_train.shape, Y_test.shape)

# set's random seed
def set_seed(seed, device):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == torch.device('cuda:0'):
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


seed = 0
dev = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Using Device: ", dev)
set_seed(seed, dev)

X_train = torch.Tensor(X_train)
Y_train = torch.Tensor(Y_train)

X_test = torch.Tensor(X_test)
Y_test = torch.Tensor(Y_test)

print("X_train.shape : {} Y_train.shape : {}".format(X_train.shape, Y_train.shape))

# converting it to Tensor datasets
train_dataset = TensorDataset(X_train, Y_train)
val_dataset = TensorDataset(X_test, Y_test)

train_batch_size = 256
val_batch_size = 256
epochs = 50

# init dataloaders
train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)

data_loaders = {"train": train_loader, "test": val_loader}

kwargs = {
    'embd_sizes': list(zip(
                            [df[c].nunique() for c in cat], # number of unique values for a feature
                            np.sqrt([df[c].nunique() for c in cat]).astype(np.int) # size of embedding
                        )),
    'sz_hidden_layers': [10, 10],
    'output_layer_sz': 1,
    'emb_layer_drop': 0.5,
    'hidden_layer_drops': [0.5, 0.5, 0.5],
    'use_bn': False,
    'y_range': None
}

pprint.pprint(kwargs)

model = EntityEmbedding(**kwargs)
model.to(dev)

learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer.zero_grad()

for epoch in range(epochs):
    optimizer.zero_grad()

    tic = time.time()
    for phase in ['train', 'test']:
        running_loss = 0.0
        if phase == 'train':
            model.train(True)
        else:
            model.train(False)

        # Iterate over data.
        tic = time.time()
        for X, Y in data_loaders[phase]:

            if str(dev) == 'cuda:0':
                X = X.type(torch.long).cuda()
                Y = Y.type(torch.float).cuda()

            Y_pred = model(X)
            loss = F.mse_loss(Y_pred, Y)

            running_loss += loss.item()

            if phase == 'train':
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
        toc = time.time()

        print("Epoch {}/{}\tPhase: {}\tLoss: {:.6f}\tTE: {:.3f}".format(epoch + 1, epochs, phase,
                                                            running_loss / len(data_loaders[phase]), toc-tic))

print("Training Over \nHere is the embeddings: \n")