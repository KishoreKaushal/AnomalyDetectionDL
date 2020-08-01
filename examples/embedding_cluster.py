import sys

sys.path.insert(0, "../")

import numpy as np
import pandas as pd
import torch
from OutlierDetection.Embedding import EntityEmbedding
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import pprint

train_data_path = "~/dataset/train_24072020.pkl"
test_data_path = "~/dataset/test_24072020.pkl"

df_train = pd.read_pickle(train_data_path)
df_test = pd.read_pickle(test_data_path)

# preprocessing
day_dict = {
    'SUN' : 1,
    'MON' : 2,
    'TUE' : 3,
    'WED' : 4,
    'THU' : 5,
    'FRI' : 6,
    'SAT' : 7
}

def pre_process(df, inplace=True):
    if not inplace:
        df = df.copy()
    df['APIKEY'] = df['APIKEY'].apply(lambda x: x.replace("APIKEY", "")).astype(int)
    df['TIMEBIN'] = df['TIMEBIN'].apply(lambda x: x.replace("BIN", "")).astype(int)
    df['LABEL'] = df['LABEL'].apply(lambda x: 0 if x == 'NOT ANAMOLY' else 1)
    df['DAY'] = df['DAY'].apply(lambda x : day_dict[x]).astype(int)
    df.drop(columns=['ANAMOLYDISTNUM'], inplace=True)
    df = df.dropna()
    return df

pre_process(df_train)
pre_process(df_test)

cat = ['APIKEY', 'DAY', 'TIMEBIN']

label_feature = ['LABEL']
non_label_features = list(set(df_train.columns) - set(label_feature))

target_feature = ['NUMREQUESTS']
non_target_feature = list(set(df_train.columns) - set(label_feature) - set(target_feature))

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

# print("Shape of Datafram : ", df.shape)
# print("NUM APIKEY: ", df['APIKEY'].nunique())
# print("NUM API: ", df['API'].nunique())
# print("NUM TIMEBIN: ", df['TIMEBIN'].nunique())

embd_sizes = np.sqrt([df_train['APIKEY'].nunique(),
                      df_train['DAY'].nunique(),
                      df_train['TIMEBIN'].nunique()]).astype(np.int)


X_train = torch.Tensor(df_train[non_target_feature].values)
Y_train = torch.Tensor(df_train[target_feature].values)

X_test = torch.Tensor(df_test[non_target_feature].values)
Y_test = torch.Tensor(df_test[target_feature].values)


dev = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Using Device: ", dev)

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
    'embd_sizes': list(zip([df_train[c].nunique() for c in cat], embd_sizes)),
    'sz_hidden_layers': [10, 10],
    'output_layer_sz': 1,
    'emb_layer_drop': 0.5,
    'hidden_layer_drops': [0.5, 0.5, 0.5],
    'use_bn': False,
    'y_range': None
}

model = EntityEmbedding(**kwargs)
model.to(dev)

learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer.zero_grad()

for epoch in range(epochs):
    optimizer.zero_grad()

    for phase in ['train', 'test']:
        running_loss = 0.0
        if phase == 'train':
            model.train(True)
        else:
            model.train(False)

        # Iterate over data.
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

        print("Epoch {}/{}\tPhase: {}\tLoss: {:.6f}".format(epoch + 1, epochs, phase,
                                                            running_loss / len(data_loaders[phase])))

print("Training Over \nHere is the embeddings: \n")

pprint.pprint(kwargs)
pprint.pprint(model.get_all_feature_embedding(True))
