'''
Description: Training script for SUPREME adapted to MultiREM project 

TODO: 
- Add mlflow for run tracking 

'''


'''
Description: Training script for SUPREME adapted to MultiREM project 

TODO: 
- Add mlflow for run tracking 

'''



from lib.model import Net, train_epoch, validate_epoch
# from lib.config import Config 

import time
import os
import re
import itertools
import pickle5 as pickle
import argparse
import errno
import warnings
import logging

import pandas as pd
import numpy as np

from sklearn.metrics import f1_score, accuracy_score
import statistics
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split, RandomizedSearchCV, GridSearchCV

import torch
from torch_geometric.data import Data

import plotly.express as px

# TODO: rpy2 doesn't work on m1 macs -> need to find a way around this 
# import rpy2
# import rpy2.robjects as robjects
# from rpy2.robjects.packages import importr
# utils = importr('utils')
# rFerns = importr('rFerns')
# Boruta = importr('Boruta')
# pracma = importr('pracma')
# dplyr = importr('dplyr')


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("RUNNING MULTIREM")

# GLOBALS
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else (torch.device('mps') if torch.has_mps else 'cpu'))
DEVICE = torch.device('cpu')
NETWORKS = ['exp', 'cna']
DATA_DIR = '../data/sample_data'
OUTPUT_DIR = './output'

NODE_FEATURE_SELECTION = False # Reduce dimensionality of node vectors (TODO make this part of config)
NODE_NUM_FEATURES = [50] * len(NETWORKS)
NUM_BORUTA_RUNS = 100 # Number of Boruta runs to perform for feature selection (TODO make this part of config)

MAX_EPOCHS = 500
MIN_EPOCHS = 200
PATIENCE = 30

HIDDEN_LAYER_SIZES = [50, 100, 200]
LEARNING_RATES = [0.01, 0.001, 0.0001]

os.makedirs(OUTPUT_DIR, exist_ok=True)



# FUNCTIONS
def get_data(dir, networks): 
    labels = pd.read_pickle(os.path.join(dir, 'labels.pkl'))
    nodes = {}
    edges = {}
    for n in networks: 
        nodes[n] = pd.read_pickle(os.path.join(dir, f'{n}.pkl'))
        edges[n] = pd.read_pickle(os.path.join(dir, f'edges_{n}.pkl'))

    return labels, nodes, edges

    

def select_node_features(feat, num_features, boruta_runs, device): 
    # Boruta feature selection to reduce dimensionality of node vectors 
    # TODO: This doesn't work on ARM for M1 macs because of some R dependency -> migrate to python

    feat_flat = [item for sublist in feat.values.tolist() for item in sublist]
    feat_temp = robjects.FloatVector(feat_flat)
    robjects.globalenv['feat_matrix'] = robjects.r('matrix')(feat_temp)
    robjects.globalenv['feat_x'] = robjects.IntVector(feat.shape)
    robjects.globalenv['labels_vector'] = robjects.IntVector(labels.tolist())
    robjects.globalenv['top'] = num_features
    robjects.globalenv['maxBorutaRuns'] = boruta_runs
    robjects.r('''
        require(rFerns)
        require(Boruta)
        labels_vector = as.factor(labels_vector)
        feat_matrix <- Reshape(feat_matrix, feat_x[1])
        feat_data = data.frame(feat_matrix)
        colnames(feat_data) <- 1:feat_x[2]
        feat_data <- feat_data %>%
            mutate('Labels' = labels_vector)
        boruta.train <- Boruta(feat_data$Labels ~ ., data= feat_data, doTrace = 0, getImp=getImpFerns, holdHistory = T, maxRuns = maxBorutaRuns)
        thr = sort(attStats(boruta.train)$medianImp, decreasing = T)[top]
        boruta_signif = rownames(attStats(boruta.train)[attStats(boruta.train)$medianImp >= thr,])
            ''')
    boruta_signif = robjects.globalenv['boruta_signif']
    robjects.r.rm("feat_matrix")
    robjects.r.rm("labels_vector")
    robjects.r.rm("feat_data")
    robjects.r.rm("boruta_signif")
    robjects.r.rm("thr")
    topx = []
    for index in boruta_signif:
        t_index=re.sub("`","",index)
        topx.append((np.array(feat.values).T)[int(t_index)-1])
    topx = np.array(topx)
    values = torch.tensor(topx.T, device=device)

    return values

def train_model(network_name, X_nodes_cat, labels, edge_index, train_valid_idx, test_idx, hid_sizes, learning_rates): 

    best_valid_loss = np.Inf

    # For each hyperparam combo
    for hid_size, learning_rate in itertools.product(hid_sizes, learning_rates):  
        logger.info(f"Training model with hidden size {hid_size} and learning rate {learning_rate}")

        # Build a graph where node features have all node feature datatypes concatenated
        #  - The edges are still just for this network 
        data = Data(x=X_nodes_cat, 
                    edge_index=torch.tensor(edge_index[edge_index.columns[0:2]].transpose().values, device=DEVICE).long(),
                    edge_attr=torch.tensor(edge_index[edge_index.columns[2]].transpose().values, device=DEVICE).float(), 
                    y=labels) 

        # Split data into training and validation sets
        X = data.x[train_valid_idx]
        y = data.y[train_valid_idx]

        # For each split 
        rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=1)
        min_valid_losses = []
        for train_part, valid_part in rskf.split(X, y):
            train_idx = train_valid_idx[train_part]
            valid_idx = train_valid_idx[valid_part]

            # Get train/valid split 
            data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            data.train_mask[train_idx] = True
            data.valid_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            data.valid_mask[valid_idx] = True

            # Build model 
            model = Net(in_size=data.x.shape[1], hid_size=hid_size, out_size=num_classes).to(DEVICE)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            criterion = torch.nn.CrossEntropyLoss()

            # Train model 
            min_valid_loss = np.Inf
            patience_count = 0

            for epoch in range(MAX_EPOCHS):
                emb = train_epoch(model, data, optimizer, criterion)
                curr_valid_loss, emb = validate_epoch(model, data, criterion)

                # Save the min valid loss across all epochs 
                if curr_valid_loss < min_valid_loss:
                    min_valid_loss = curr_valid_loss
                    patience_count = 0
                else:
                    patience_count += 1

                # If model isn't learning anything, break
                if epoch >= MIN_EPOCHS and patience_count >= PATIENCE:
                    break
            
            min_valid_losses.append(min_valid_loss.item())

        curr_min_valid_loss_overall = np.mean(min_valid_losses)            
        if curr_min_valid_loss_overall < best_valid_loss:
            best_valid_loss = curr_min_valid_loss_overall
            best_emb_lr = learning_rate
            best_emb_hs = hid_size


    # Retrain the model on the entire training set with the best hyperparams
    logger.info(f"Retraining model on full train/valid with hidden size {best_emb_hs} and learning rate {best_emb_lr}")
    data = Data(x=X_nodes_cat, edge_index=torch.tensor(edge_index[edge_index.columns[0:2]].transpose().values, device=DEVICE).long(),
                edge_attr=torch.tensor(edge_index[edge_index.columns[2]].transpose().values, device=DEVICE).float(), y=labels) 
    X = data.x[train_valid_idx]
    y = data.y[train_valid_idx]

    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.train_mask[train_valid_idx] = True
    data.valid_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.valid_mask[test_idx] = True

    model = Net(in_size=data.x.shape[1], hid_size=best_emb_hs, out_size=num_classes).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=best_emb_lr)
    criterion = torch.nn.CrossEntropyLoss()

    selected_emb = None
    min_valid_loss = np.Inf
    patience_count = 0
    history = []
                
    for epoch in range(MAX_EPOCHS):
        emb = train_epoch(model, data, optimizer, criterion)
        curr_valid_loss, emb = validate_epoch(model, data, criterion)
        history.append(curr_valid_loss)

        if curr_valid_loss < min_valid_loss:
            min_valid_loss = curr_valid_loss
            patience_count = 0
            selected_emb = emb
        else:
            patience_count += 1

        if epoch >= MIN_EPOCHS and patience_count >= PATIENCE:
            break

    logger.info(f"Final test loss: {min_valid_loss}")
    px.line(history).write_image(os.path.join(OUTPUT_DIR, f"{network_name}_gcn_history.jpg"))

    # Save the embeddings 
    emb_file = os.path.join(OUTPUT_DIR, f"{network_name}_embeddings.pkl")
    with open(emb_file, 'wb') as f:
        pickle.dump(selected_emb, f)
        pd.DataFrame(selected_emb).to_csv(emb_file[:-4] + '.csv')

    return selected_emb


if __name__ == '__main__':

    labels, nodes, edges = get_data(DATA_DIR, NETWORKS)
    num_classes = torch.unique(labels).shape[0]
    logger.info(f'Number of classes: {num_classes}')

    train_valid_idx, test_idx = train_test_split(np.arange(len(labels)), test_size=0.20, shuffle=True, stratify=labels, random_state=42)

    # Run feature selection if desired 
    if NODE_FEATURE_SELECTION:
        X_nodes = {}
        for n in NETWORKS:
            # TODO: This doesn't work 
            X_nodes[n] = select_node_features(nodes[n], NODE_NUM_FEATURES[NETWORKS.index(n)], NUM_BORUTA_RUNS, DEVICE)
    else:
        X_nodes = nodes

    X_nodes_cat = torch.cat([torch.tensor(v.values).float() for v in X_nodes.values()], dim=1)


    # For each network 
    for n in NETWORKS: 
        logger.info(f"Training model for network {n}")
        edge_index = edges[n]

        # Train model
        emb = train_model(n, X_nodes_cat, labels, edge_index, train_valid_idx, test_idx, HIDDEN_LAYER_SIZES, LEARNING_RATES)
    