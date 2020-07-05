import sys
sys.path.insert(0, "../")

import numpy as np
import pandas as pd
import datetime
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
import torch
from OutlierDetection.Embedding import EntityEmbedding
import torch.nn.functional as F
from torch.utils.data import TensorDataset,DataLoader

# set's random seed
def set_seed(seed, device):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == torch.device('cuda'):
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

seed = 33
dev = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Using Device: ", dev)
set_seed(seed, dev)

df = pd.read_pickle('../data/dataset.pkl')
cat = ['APIKEY', 'API', 'TIMEBIN']
target = ['NUMREQUESTS']

df_cat = df[cat]

enc = OrdinalEncoder()
df_cat = enc.fit_transform(df_cat)
# splitting the dataset into train and validation
df_train, df_val, df_cat_train, df_cat_val = train_test_split(df, df_cat, test_size=0.2, random_state=seed)

# converting it into tensors
X_train, X_val = torch.Tensor(df_cat_train), torch.Tensor(df_cat_val)
Y_train, Y_val = torch.Tensor(df_train[target].values), torch.Tensor(df_val[target].values)

# converting it to Tensor datasets
train_dataset = TensorDataset(X_train, Y_train)
val_dataset = TensorDataset(X_val, Y_val)

kwargs = {
    'embd_sizes' : list(zip([df[c].nunique() for c in cat], [3, 4, 5])),
    'sz_hidden_layers' : [10, 10],
    'output_layer_sz' : 1,
    'emb_layer_drop' : 0.5,
    'hidden_layer_drops' : [0.5, 0.5, 0.5],
    'use_bn' : False,
    'y_range' : None
}

model = EntityEmbedding(**kwargs)
model.to(dev)

batch_size = 128
epochs = 50

# init dataloaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer.zero_grad()

for epoch in range(epochs):
    print("Epoch {}/{}".format(epoch+1, epochs))
    model.train()
    t1 = datetime.datetime.now()
    
    train_loss = 0
    for X,Y in train_loader:
        X = X.type(torch.long).to(dev)
        Y = Y.type(torch.long).to(dev)
        Y_pred = model(X)
        loss = F.mse_loss(Y_pred, Y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss = train_loss + loss.item()
    
    val_loss = 0
    with torch.no_grad():
        model.eval()
        for X,Y in val_loader:
            X = X.type(torch.long).to(dev)
            Y = Y.type(torch.long).to(dev)
            Y_pred = model(X)
            loss = F.mse_loss(Y_pred, Y)
            val_loss = val_loss + loss.item()
    
    train_loss = round(train_loss/len(train_loader), 5)
    val_loss = round(val_loss/len(val_loader), 5)
    t2 = datetime.datetime.now()
    timetaken = round((t2-t1).total_seconds())
    print("Seconds = {} train loss = {} val loss = {}".format(timetaken, train_loss, val_loss))