'''
Description: Run prediction on some graphs 


'''

from lib.model import Net, predict

import os
import json
import pickle
import logging

import pandas as pd

import torch
from torch_geometric.data import Data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# GLOBALS 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NETWORKS = ['exp', 'cna']
DATA_DIR = '../data/sample_data'
OUTPUT_DIR = './output'

NODE_FEATURE_SELECTION = False # Reduce dimensionality of node vectors (TODO make this part of config)
NODE_NUM_FEATURES = [50] * len(NETWORKS)
NUM_BORUTA_RUNS = 100 # Number of Boruta runs to perform for feature selection (TODO make this part of config)

ADD_RAW_FEATURES = True
CLASSIFIER = "MLP"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# FUNCTIONS 
def get_data(dir, networks): 
    nodes = {}
    edges = {}
    for n in networks: 
        nodes[n] = pd.read_pickle(os.path.join(dir, f'{n}.pkl'))
        edges[n] = pd.read_pickle(os.path.join(dir, f'edges_{n}.pkl'))

    return nodes, edges


if __name__ == "__main__":

    # Load models 
    models = {}
    hyperparams = {}
    for n in NETWORKS: 

        with open(os.path.join(OUTPUT_DIR, f"{n}_gcn_params.json"), 'r') as f:
            hyperparams[n] = json.load(f)

        model = Net(in_size=hyperparams[n]['input_size'], hid_size=hyperparams[n]['hidden_size'], out_size=hyperparams[n]['output_size']).to(DEVICE)
        model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, f'{n}_gcn_model.pt')))
        models[n] = model

    # Load pickled classifier
    with open(os.path.join(OUTPUT_DIR, f"{CLASSIFIER}_classifier.pkl"), 'rb') as f:
        models['classifier'] = pickle.load(f)

    nodes, edges = get_data(DATA_DIR, NETWORKS)

    # Run feature selection if desired 
    if NODE_FEATURE_SELECTION:
        X_nodes = {}
        for n in NETWORKS:
            # TODO: This doesn't work 
            raise NotImplementedError
            # X_nodes[n] = select_node_features(nodes[n], NODE_NUM_FEATURES[NETWORKS.index(n)], NUM_BORUTA_RUNS, DEVICE)
    else:
        X_nodes = nodes

    X_nodes_cat = torch.cat([torch.tensor(v.values).float() for v in X_nodes.values()], dim=1)

    # Get node embeddings from gcn models 
    node_embeddings = {}
    gcn_predictions = {}
    for n in NETWORKS:
        edge_index = edges[n]
        data = Data(x=X_nodes_cat, 
                        edge_index=torch.tensor(edge_index[edge_index.columns[0:2]].transpose().values, device=DEVICE).long(),
                        edge_attr=torch.tensor(edge_index[edge_index.columns[2]].transpose().values, device=DEVICE).float())
        pred, emb = predict(models[n], data)
        gcn_predictions[n] = pred
        node_embeddings[n] = emb

    # Concatenate node embeddings
    node_embeddings_concat = torch.cat([node_embeddings[n] for n in NETWORKS], axis=1)

    # Add raw features if desired
    if ADD_RAW_FEATURES: 
        X_nodes_cat_raw = torch.cat([torch.tensor(v.values).float() for v in nodes.values()], dim=1)
        if NODE_FEATURE_SELECTION: 
            raise NotImplementedError
            # X_nodes_cat_raw = select_node_features(X_nodes_cat_raw, NODE_NUM_FEATURES[NETWORKS.index(n)], NUM_BORUTA_RUNS, DEVICE)

        node_embeddings_concat = torch.cat([node_embeddings_concat, X_nodes_cat_raw], dim=1)
        logger.info(f"Final node embeddings shape with raw features: {node_embeddings_concat.shape}")

    X = node_embeddings_concat.cpu().detach().numpy()

    # Run classifier
    y_pred = models['classifier'].predict(X)

    # Save predictions
    with open(os.path.join(OUTPUT_DIR, 'predictions.pkl'), 'wb') as f:
        pickle.dump(y_pred, f)
