import sys
sys.path.insert(0, "../")

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import torch
from OutlierDetection.Embedding import EntityEmbedding
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset,DataLoader


df = pd.read_pickle('../data/dataset.pkl')
cat = ['APIKEY', 'API', 'TIMEBIN']
target = ['NUMREQUESTS']

print("Shape of Datafram : ", df.shape)
print("NUM APIKEY: ", df['APIKEY'].nunique())
print("NUM API: ", df['API'].nunique())
print("NUM TIMEBIN: ", df['TIMEBIN'].nunique())

embd_sizes = np.sqrt([df['APIKEY'].nunique(), df['API'].nunique(), df['TIMEBIN'].nunique()]).astype(np.int)

df_cat = df[cat]

enc = OrdinalEncoder()
df_cat = enc.fit_transform(df_cat)

X_train, X_test, Y_train, Y_test = train_test_split(df_cat, np.log1p(df[target].values), test_size=0.2)

X_train = torch.Tensor(X_train)
Y_train = torch.Tensor(Y_train)

X_test = torch.Tensor(X_test)
Y_test = torch.Tensor(Y_test)

#
# X = torch.Tensor(df_cat)
# Y = torch.Tensor(df[target].values)

dev = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Using Device: ", dev)

# if str(dev) == 'cuda:0':
#     X_train = X_train.type(torch.long).cuda()
#     Y_train = Y_train.type(torch.float).cuda()
#
#     X_test = X_test.type(torch.long).cuda()
#     Y_test = Y_test.type(torch.float).cuda()

# converting it to Tensor datasets
train_dataset = TensorDataset(X_train, Y_train)
val_dataset = TensorDataset(X_test, Y_test)

train_batch_size = 128
val_batch_size = 128
epochs = 50

# init dataloaders
train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)

data_loaders = {"train": train_loader, "test": val_loader}

kwargs = {
    'embd_sizes' : list(zip([df[c].nunique() for c in cat], embd_sizes)),
    'sz_hidden_layers' : [10, 10],
    'output_layer_sz' : 1,
    'emb_layer_drop' : 0.5,
    'hidden_layer_drops' : [0.5, 0.5, 0.5],
    'use_bn' : False,
    'y_range' : None
}

model = EntityEmbedding(**kwargs)
model.to(dev)

learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)



for epoch in range(epochs):
    optimizer.zero_grad()


    for phase in ['train', 'test']:
        running_loss = 0.0
        if phase == 'train':
            model.train(True)
        else:
            model.train(False)

        # Iterate over data.
        for X,Y in data_loaders[phase]:
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

        print("Epoch {}/{}\tPhase: {}\tLoss: {:.6f}".format(epoch + 1, epochs, phase,
                                                            running_loss/len(data_loaders[phase])))