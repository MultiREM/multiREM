'''
Description: Training script for SUPREME adapted to MultiREM project 

TODO: 
- Add mlflow for run tracking 
- Fix feature selection so it's not using this wack R code 
- Add inference step 
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


ADD_RAW_FEATURES = True
CLASSIFIER = "MLP"

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


def get_classifier(classifier, X_train, y_train): 
    logger.info(f"Training classifier {classifier}")
    if classifier == 'MLP':
        params = {'hidden_layer_sizes': [(16,), (32,),(64,),(128,),(256,),(512,), (32, 32), (64, 32), (128, 32), (256, 32), (512, 32)]}
        search = RandomizedSearchCV(estimator = MLPClassifier(solver = 'adam', activation = 'relu', early_stopping = True), 
                                    return_train_score = True, scoring = 'f1_macro', 
                                    param_distributions = params, cv = 4, n_iter = 10, verbose = 0)
        search.fit(X_train, y_train)
        model = MLPClassifier(solver = 'adam', activation = 'relu', early_stopping = True,
                                hidden_layer_sizes = search.best_params_['hidden_layer_sizes'])
        
    elif classifier == 'XGBoost':
        params = {'reg_alpha':range(0,6,1), 'reg_lambda':range(1,5,1),
                    'learning_rate':[0, 0.001, 0.01, 1]}
        fit_params = {'early_stopping_rounds': 10,
                        'eval_metric': 'mlogloss',
                        'eval_set': [(X_train, y_train)]}
                
        search = RandomizedSearchCV(estimator = XGBClassifier(use_label_encoder=False, n_estimators = 1000, 
                                                                    fit_params = fit_params, objective="multi:softprob", eval_metric = "mlogloss", 
                                                                    verbosity = 0), return_train_score = True, scoring = 'f1_macro',
                                        param_distributions = params, cv = 4, n_iter = 10, verbose = 0)
        
        search.fit(X_train, y_train)
        
        model = XGBClassifier(use_label_encoder=False, objective="multi:softprob", eval_metric = "mlogloss", verbosity = 0,
                                n_estimators = 1000, fit_params = fit_params,
                                reg_alpha = search.best_params_['reg_alpha'],
                                reg_lambda = search.best_params_['reg_lambda'],
                                learning_rate = search.best_params_['learning_rate'])
                            
    elif classifier == 'RF':
        max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
        max_depth.append(None)
        params = {'n_estimators': [int(x) for x in np.linspace(start = 200, stop = 2000, num = 100)]}
        search = RandomizedSearchCV(estimator = RandomForestClassifier(), return_train_score = True,
                                    scoring = 'f1_macro', param_distributions = params, cv=4,  n_iter = 10, verbose = 0)
        search.fit(X_train, y_train)
        model=RandomForestClassifier(n_estimators = search.best_params_['n_estimators'])

    elif classifier == 'SVM':
        params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                    'gamma': [1, 0.1, 0.01, 0.001]}
        search = RandomizedSearchCV(SVC(), return_train_score = True,
                                    scoring = 'f1_macro', param_distributions = params, cv=4, n_iter = 10, verbose = 0)
        search.fit(X_train, y_train)
        model=SVC(C = search.best_params_['C'],
                    gamma = search.best_params_['gamma'])

    logger.info(f'selected parameters = {search.best_params_}')
    return model

def evaluate_classifier(model, X_train, y_train, X_test, y_test): 

    metrics = {
        'train_acc': [],
        'train_wf1': [],
        'train_mf1': [],
        'test_acc': [],
        'test_wf1': [],
        'test_mf1': []
    }

    # Run classifier 10 times and average results
    for _ in range(10):
        model.fit(X_train,y_train)
        predictions = model.predict(X_test)
        y_pred = [round(value) for value in predictions]
        # preds = model.predict(pd.DataFrame(data.x.numpy()))
        tr_predictions = model.predict(X_train)
        tr_pred = [round(value) for value in tr_predictions]

        metrics['train_acc'].append(round(accuracy_score(y_train, tr_pred), 3))
        metrics['train_wf1'].append(round(f1_score(y_train, tr_pred, average='weighted'), 3))
        metrics['train_mf1'].append(round(f1_score(y_train, tr_pred, average='macro'), 3))

        metrics['test_acc'].append(round(accuracy_score(y_test, y_pred), 3))
        metrics['test_wf1'].append(round(f1_score(y_test, y_pred, average='weighted'), 3))
        metrics['test_mf1'].append(round(f1_score(y_test, y_pred, average='macro'), 3))

    logger.info("\n".join([f"{k}: {round(np.mean(v), 3)}+-{round(np.std(v), 3)}" for k, v in metrics.items()]))

    return metrics 

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

    node_embeddings = {}
    # For each network generate GCN node embeddings
    for n in NETWORKS: 
        logger.info(f"Training model for network {n}")
        edge_index = edges[n]

        # Train model
        emb = train_model(n, X_nodes_cat, labels, edge_index, train_valid_idx, test_idx, HIDDEN_LAYER_SIZES, LEARNING_RATES)
        node_embeddings[n] = emb

    # Use embeddings from all networks for classifier 
    # Concat embeddings from each network 
    node_embeddings_concat = torch.cat([node_embeddings[n] for n in NETWORKS], axis=1)
    logger.info(f"Final node embeddings shape: {node_embeddings_concat.shape}")

    # Add raw features if desired
    if ADD_RAW_FEATURES: 
        X_nodes_cat_raw = torch.cat([torch.tensor(v.values).float() for v in nodes.values()], dim=1)
        if NODE_FEATURE_SELECTION: 
            X_nodes_cat_raw = select_node_features(X_nodes_cat_raw, NODE_NUM_FEATURES[NETWORKS.index(n)], NUM_BORUTA_RUNS, DEVICE)

        node_embeddings_concat = torch.cat([node_embeddings_concat, X_nodes_cat_raw], dim=1)
        logger.info(f"Final node embeddings shape with raw features: {node_embeddings_concat.shape}")

    data = Data(x=node_embeddings_concat, y=labels)

    # Split data into training and validation sets
    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.train_mask[train_valid_idx] = True
    data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.test_mask[test_idx] = True

    X_train = pd.DataFrame(data.x[data.train_mask])
    y_train = data.y[data.train_mask].numpy()
    X_test = pd.DataFrame(data.x[data.test_mask])
    y_test = data.y[data.test_mask].numpy()

    logger.info(f"Shape: {X_train.shape}, {y_train.shape}, {X_test.shape}, {y_test.shape}")

    model = get_classifier(CLASSIFIER, X_train, y_train)
    metrics = evaluate_classifier(model, X_train, y_train, X_test, y_test)

