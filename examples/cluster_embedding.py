from datetime import datetime
import sys

sys.path.insert(0, "../")

import numpy as np
import torch
from scipy.spatial import distance_matrix
from OutlierDetection.Embedding import EntityEmbedding

import pprint
import time
import os

import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

saved_model_path = "./saved_model/train_01082020.pkl_entity_embedding_04_08_2020_16_35_33.pt"

model = torch.load(saved_model_path)
assert isinstance(model, EntityEmbedding)

all_feature_embd_dict = model.get_all_feature_embedding()


feature_name_present = 'embeddingFeature' in model.meta.keys()

# generating a list of embeddings for each features
for feature_idx, this_feature_embd_dict in all_feature_embd_dict.items():
    this_feature_embedding = []
    # get feature name if exists
    feature_name = str(feature_idx)
    if feature_name_present:
        feature_name = model.meta['embeddingFeature'][feature_idx]

    for k, v in this_feature_embd_dict.items():
        this_feature_embedding.append(v)

    this_feature_embedding = np.array(this_feature_embedding)
    dist_mat = distance_matrix(this_feature_embedding, this_feature_embedding)
    # Draw the full plot
    cluster_grid = sns.clustermap(dist_mat, center=0, cmap="vlag",
                                   row_cluster=True, col_cluster=True,
                                   linewidths=.75, figsize=(20, 20))

    cluster_grid.savefig("cluster_grid_{}.png".format(feature_name))
