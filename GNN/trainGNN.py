import sys
sys.settrace

import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv, TransformerConv
from torch_geometric.nn import global_mean_pool

from torch_geometric.data import Data, DataLoader, InMemoryDataset, download_url
import makePYGdata as mpd
import importlib
importlib.reload(mpd)

some_data = mpd.PTC_Trees_Dataset(root='data')
loader = DataLoader(some_data, batch_size=10)

pyg_dataset = loader.dataset

train_dataset = pyg_dataset[:10]
test_dataset = pyg_dataset[10:]



class GNN(torch.nn.Module):
    def __init__(self, input_size, hidden_channels, out_channels, conv, conv_params={}):
        super(GNN, self).__init__()
        torch.manual_seed(12345)
        
        self.conv1 = conv(input_size, hidden_channels, **conv_params)
        
        self.conv2 = conv(hidden_channels, hidden_channels, **conv_params)
        
        self.lin = Linear(hidden_channels, out_channels)
    
    def forward(self, x, edge_index, batch = None,  edge_col = None):
        
        # Node embedding 
        x = self.conv1(x, edge_index, edge_col)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_col)
        
        # Readout layer
        batch = torch.zeros(data.x.shape[0],dtype=int) if batch is None else batch
        x = global_mean_pool(x, batch)
        
        # Final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
    
        return x
    
model = GNN(pyg_dataset.num_features, 16, pyg_dataset.num_classes)



data = train_dataset.data

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(2):
    pred = model(data.x, data.edge_index)
    
    print(pred)
    
#     loss = F.cross_entropy(pred[data.train_mask], data.y[data.train_mask])

#     # Backpropagation
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()