import torch
import torch.nn as nn
import torch.nn.functional as F


class EntityEmbedding(nn.Module):
    """
        Parameters
        ----------
        embd_sizes : list of int
            Specify the list of sizes of the embedding vector
            for each categorical feature.

        sz_hidden_layers : list of int
            Specify the size of the hidden linear layers

        output_layer_sz : int
            Size of the output layers

        emb_layer_drop : float
            Dropout applied to the output of the embedding

        hidden_layer_drops : list of float
            List of dropout applied to the hidden layers

        use_bn : bool
            True, for batch normalization

        y_range : 2-tuple (min_y, max_y), optional default None
            Range of the output variable.
    """

    def __init__(self, embd_sizes, sz_hidden_layers, output_layer_sz,
                 emb_layer_drop, hidden_layer_drops, use_bn=False, y_range=None):
        super(EntityEmbedding, self).__init__()

        self.embd_sizes = embd_sizes
        self.use_bn = use_bn
        self.y_range = y_range

        self.embds = nn.ModuleList([
            nn.Embedding(num_embeddings=c, embedding_dim=s) for c, s in self.embd_sizes
        ])

        for embd in self.embds:
            embd.weight.data.uniform_(-1, 1)

        # size of the vector after concatenating all the embedding layer
        conc_embd_size = sum(e.embedding_dim for e in self.embds)

        # linear layers followed by embedding layers
        sz_hidden_layers = [conc_embd_size] + sz_hidden_layers
        self.linear_layers = nn.ModuleList([
            nn.Linear(sz_hidden_layers[i], sz_hidden_layers[i + 1])
            for i in range(len(sz_hidden_layers) - 1)
        ])

        # batch normalization layers after each linear layers
        self.batch_norm_layers = nn.ModuleList([
            nn.BatchNorm1d(sz) for sz in sz_hidden_layers[1:]
        ])

        # initializing hidden layers
        for out in self.linear_layers:
            nn.init.kaiming_normal_(out.weight.data)

        # initializing output layer
        self.output_layer = nn.Linear(sz_hidden_layers[-1], output_layer_sz)
        nn.init.kaiming_normal_(self.output_layer.weight.data)

        self.emb_drop = nn.Dropout(emb_layer_drop)

        self.hidden_dropout_layers = nn.ModuleList([
            nn.Dropout(drop) for drop in hidden_layer_drops
        ])

    def forward(self, input):
        """
            Parameters
            ----------
            input:  list of inputs => (num_inputance, num_categorical_features)

            Return
            ------
            The output of the forward propagation in the network.
        """
        x = input

        # print("INPUT : ", input)
        # temp = []
        # for i, e in enumerate(self.embds):
        #     print("i : ", i)
        #     import pdb;pdb.set_trace()
        #     temp.append(e(x[:, i]))


        x = [e(x[:, i]) for i, e in enumerate(self.embds)]

        # concatenate all embeddings
        x = torch.cat(x, 1)

        # dropout for embedding layers
        x = self.emb_drop(x)

        for linear, batch_norm, dropout in zip(self.linear_layers,
                                               self.batch_norm_layers,
                                               self.hidden_dropout_layers):
            x = F.relu(linear(x))
            if self.use_bn:
                x = batch_norm(x)
            x = dropout(x)

        x = self.output_layer(x)

        if self.y_range:
            x = F.sigmoid(x)
            x = x * (self.y_range[1] - self.y_range[0])
            x = x + self.y_range[0]

        return x

    def get_embedding_for_x(self, x):
        return [e(x[i]) for i, e in enumerate(self.embds)]

    def get_feature_embedding(self, idx, feature):
        return self.embds[feature](idx)

    def get_all_feature_embedding(self, isCuda = False):

        embeddings_dict = dict()

        for feature, (cardinality, _) in enumerate(self.embd_sizes):
            input = torch.LongTensor(list(range(cardinality)))

            embeddings_dict[feature] = dict()

            if isCuda:
                input = input.cuda()

            feature_embd = self.embds[feature](input)

            for i, embd in enumerate(feature_embd):
                embeddings_dict[feature][i] = embd

        return embeddings_dict