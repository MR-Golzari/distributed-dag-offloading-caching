import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
import random
from deap import base, creator, tools, algorithms

class GATEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(GATEncoder, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=8, dropout=0.6)
        self.conv2 = GATConv(hidden_channels * 8, hidden_channels, heads=1, concat=True, dropout=0.6)

    def forward(self, x, edge_index):
        #x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        #x = F.elu(x)
        #x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class GATDecoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super(GATDecoder, self).__init__()
        self.conv1 = GATConv(hidden_channels, hidden_channels * 8, heads=1, concat=True, dropout=0.6)
        self.conv2 = GATConv(hidden_channels * 8, out_channels, heads=1, dropout=0.6)

    def forward(self, x, edge_index):
        #x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        #x = F.elu(x)
        #x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class GATAutoencoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GATAutoencoder, self).__init__()
        self.encoder = GATEncoder(in_channels, hidden_channels)
        self.decoder = GATDecoder(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        encoded = self.encoder(x, edge_index)
        decoded = self.decoder(encoded, edge_index)
        return decoded
    
    def GAT_encode(self, data):
        x, edge_index = data.x, data.edge_index
        encoded = self.encoder(x, edge_index)
        return encoded
class GAT():
    def __init__(self,users) -> None:
        self.data={}
        for i,u in users.items():

            self.data[i]=from_networkx(u.DAG)
            x=[[0,0,0,0]]
            for t in u.tasks_init.values():
                x.append([t.cpu_cycle*1e-6/u.max_cpu_cycles, t.input_data_length/u.max_data_length, t.outputlength/u.max_data_length, t.service/u.numberofservices])
            self.data[i].x=torch.tensor(x)
        

        # Create the graph data object
        # data = Data(x=x, edge_index=edge_index)

        # Define the model
        self.model = GATAutoencoder(in_channels=4, hidden_channels=16, out_channels=4)

        # Define the optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.005, weight_decay=5e-4)

        # Training loop
        self.model.train()
        for epoch in range(200):
            for data in self.data.values():
                optimizer.zero_grad()
                out = self.model(data)
                loss = F.mse_loss(out, data.x)  # Reconstruction loss
                loss.backward()
                optimizer.step()
            print(f'Epoch {epoch+1}, Loss: {loss.item()}')

        # Test the model
        self.model.eval()
        for data in self.data.values():
            with torch.no_grad():
                out = self.model(data)
                print(f'Original features: {data.x}')
                print(f'Reconstructed features: {out}')
        
    def encode(self,data):
        encoded_data  = self.model.GAT_encode(data)
        return encoded_data
 

