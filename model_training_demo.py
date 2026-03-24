#!/usr/bin/env python3
"""Model Training Demo for JittorGeometric"""

import jittor as jt
from jittor import nn
from jittor_geometric.datasets import Planetoid
from jittor_geometric.nn.conv import GCNConv, GATConv, SAGEConv
import time

class ModelTrainer:
    def __init__(self, model_name='GCN', dataset_name='Cora'):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.dataset = None
        self.model = None
        self.optimizer = None
        self.criterion = None
        
    def load_data(self):
        print(f"Loading dataset: {self.dataset_name}")
        self.dataset = Planetoid(root='/tmp/' + self.dataset_name, name=self.dataset_name)
        return self.dataset
    
    def create_model(self, hidden_dim=16, num_layers=2, heads=8, dropout=0.5):
        print(f"Creating model: {self.model_name}")
        
        if self.model_name == 'GCN':
            class GCN(nn.Module):
                def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
                    super().__init__()
                    self.layers = nn.ModuleList()
                    self.layers.append(GCNConv(in_channels, hidden_channels))
                    for _ in range(num_layers - 2):
                        self.layers.append(GCNConv(hidden_channels, hidden_channels))
                    self.layers.append(GCNConv(hidden_channels, out_channels))
                    self.dropout = dropout
                    self.relu = nn.ReLU()
                    
                def execute(self, data):
                    x, edge_index = data.x, data.edge_index
                    from jittor_geometric.ops import cootocsr, cootocsc
                    from jittor_geometric.nn.conv.gcn_conv import gcn_norm
                    
                    # Normalize edge indices
                    edge_index, edge_weight = gcn_norm(edge_index, None, x.size(0), improved=False, add_self_loops=True)
                    
                    # Convert to CSR and CSC formats
                    csr = cootocsr(edge_index, edge_weight, x.size(0))
                    csc = cootocsc(edge_index, edge_weight, x.size(0))
                    
                    for i, conv in enumerate(self.layers[:-1]):
                        x = conv(x, csc, csr)
                        x = self.relu(x)
                        x = nn.dropout(x, p=self.dropout)
                    x = self.layers[-1](x, csc, csr)
                    return x
            
            self.model = GCN(
                in_channels=self.dataset.num_features,
                hidden_channels=hidden_dim,
                out_channels=self.dataset.num_classes,
                num_layers=num_layers,
                dropout=dropout
            )
        elif self.model_name == 'GAT':
            class GAT(nn.Module):
                def __init__(self, in_channels, hidden_channels, out_channels, num_layers, heads, dropout):
                    super().__init__()
                    self.layers = nn.ModuleList()
                    self.layers.append(GATConv(in_channels, hidden_channels, heads=heads))
                    for _ in range(num_layers - 2):
                        self.layers.append(GATConv(hidden_channels*heads, hidden_channels, heads=heads))
                    self.layers.append(GATConv(hidden_channels*heads, out_channels, heads=1))
                    self.dropout = dropout
                    self.elu = nn.ELU()
                    
                def execute(self, data):
                    x, edge_index = data.x, data.edge_index
                    for i, conv in enumerate(self.layers[:-1]):
                        x = conv(x, edge_index)
                        x = self.elu(x)
                        x = nn.dropout(x, p=self.dropout)
                    x = self.layers[-1](x, edge_index)
                    return x
            
            self.model = GAT(
                in_channels=self.dataset.num_features,
                hidden_channels=hidden_dim,
                out_channels=self.dataset.num_classes,
                num_layers=num_layers,
                heads=heads,
                dropout=dropout
            )
        elif self.model_name == 'GraphSAGE':
            class GraphSAGE(nn.Module):
                def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
                    super().__init__()
                    self.layers = nn.ModuleList()
                    self.layers.append(SAGEConv(in_channels, hidden_channels))
                    for _ in range(num_layers - 2):
                        self.layers.append(SAGEConv(hidden_channels, hidden_channels))
                    self.layers.append(SAGEConv(hidden_channels, out_channels))
                    self.dropout = dropout
                    self.relu = nn.ReLU()
                    
                def execute(self, data):
                    x, edge_index = data.x, data.edge_index
                    for i, conv in enumerate(self.layers[:-1]):
                        x = conv(x, edge_index)
                        x = self.relu(x)
                        x = nn.dropout(x, p=self.dropout)
                    x = self.layers[-1](x, edge_index)
                    return x
            
            self.model = GraphSAGE(
                in_channels=self.dataset.num_features,
                hidden_channels=hidden_dim,
                out_channels=self.dataset.num_classes,
                num_layers=num_layers,
                dropout=dropout
            )
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
        
        self.optimizer = nn.Adam(self.model.parameters(), lr=0.01)
        self.criterion = nn.CrossEntropyLoss()
        
        return self.model
    
    def train(self, epochs=200, status_callback=None):
        print(f"\nTraining {self.model_name} on {self.dataset_name} for {epochs} epochs...")
        
        self.model.train()
        start_time = time.time()
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            out = self.model(self.dataset[0])
            loss = self.criterion(out[self.dataset[0].train_mask], self.dataset[0].y[self.dataset[0].train_mask])
            self.optimizer.backward(loss)
            self.optimizer.step()
            
            acc = self.test()
            
            # Call status callback if provided
            if status_callback:
                status_callback(epoch, loss.item(), acc)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d}, Loss: {loss.item():.4f}, Acc: {acc:.4f}")
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.2f} seconds")
        
    def test(self):
        self.model.eval()
        out = self.model(self.dataset[0])
        pred = out.argmax(dim=1)[0]
        correct = jt.zeros_like(pred)
        correct[pred == self.dataset[0].y] = 1
        correct = correct[self.dataset[0].test_mask]
        acc = correct.sum() / self.dataset[0].test_mask.sum()
        return acc.numpy()[0]
    
    def predict(self, x, edge_index):
        self.model.eval()
        return self.model(self.dataset[0])[0]

def main():
    print("JittorGeometric Model Training Demo")
    print("="*50)
    
    # Create trainer
    trainer = ModelTrainer(model_name='GCN', dataset_name='Cora')
    
    # Load data
    dataset = trainer.load_data()
    print(f"Dataset: {dataset.name}")
    print(f"Number of nodes: {dataset[0].num_nodes}")
    print(f"Number of edges: {dataset[0].num_edges}")
    print(f"Number of features: {dataset.num_features}")
    print(f"Number of classes: {dataset.num_classes}")
    
    # Create model
    trainer.create_model(hidden_dim=16, num_layers=2, dropout=0.5)
    
    # Train model
    trainer.train(epochs=200)
    
    # Test model
    acc = trainer.test()
    print(f"\nTest Accuracy: {acc:.4f}")
    
    # Example prediction
    print("\nMaking predictions...")
    out = trainer.predict(dataset[0].x, dataset[0].edge_index)
    pred = out.argmax(dim=1)[0]
    print(f"Predictions shape: {pred.shape}")
    print(f"First 10 predictions: {pred[:10]}")

if __name__ == "__main__":
    main()