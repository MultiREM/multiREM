import torch.nn.functional as F
import torch
from torch_geometric.nn import GCNConv

class Net(torch.nn.Module):
    def __init__(self, in_size=16, hid_size=8, out_size=2):
        super(Net, self).__init__()
        self.conv1 = GCNConv(in_size, hid_size)
        self.conv2 = GCNConv(hid_size, out_size)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x_emb = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x_emb)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x, x_emb


def train_epoch(model, data, optimizer, criterion): 
    model.train()
    optimizer.zero_grad()
    out, emb1 = model(data)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return emb1

def validate_epoch(model, data, criterion):
    model.eval()
    with torch.no_grad():
        out, emb2 = model(data)
        pred = out.argmax(dim=1)
        loss = criterion(out[data.valid_mask], data.y[data.valid_mask])        
    return loss, emb2

def predict(model, data): 
    model.eval()
    with torch.no_grad():
        out, emb = model(data)
        pred = out.argmax(dim=1)
    return pred, emb