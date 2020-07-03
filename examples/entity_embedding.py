import sys
sys.path.insert(0, "../")

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import torch
from OutlierDetection.Embedding import EntityEmbedding
import torch.nn.functional as F


df = pd.read_pickle('../data/sampledataset.pkl')
cat = ['APIKEY', 'API', 'TIMEBIN']
target = ['NUMREQUESTS']

df_cat = df[cat]

enc = OrdinalEncoder()
df_cat = enc.fit_transform(df_cat)

X = torch.Tensor(df_cat)
Y = torch.Tensor(df[target].values)

dev = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Using Device: ", dev)

if str(dev) == 'cuda:0':
    X = X.type(torch.long).cuda()
    Y = Y.type(torch.long).cuda()

kwargs = {
    'embd_sizes' : list(zip([df[c].nunique() for c in cat], [2, 3, 5])),
    'sz_hidden_layers' : [50, 50],
    'output_layer_sz' : 1,
    'emb_layer_drop' : 0.5,
    'hidden_layer_drops' : [0.5, 0.5, 0.5],
    'use_bn' : True,
    'y_range' : None
}

model = EntityEmbedding(**kwargs)
model.to(dev)

learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

epoch = 10
for i in range(epoch):
    print("\nEpoch: {}".format(i))

    optimizer.zero_grad()
    output = model(X)
    loss = F.mse_loss(output, Y)
    loss.backward()
    optimizer.step()
    print('Train Epoch: {} \tLoss: {:.6f}'.format(i, loss.item()))

