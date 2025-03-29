import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)  # BatchNorm for hidden_size, not 1
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size // 2)  # BatchNorm for hidden_size // 2
        self.fc3 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.batch_norm3 = nn.BatchNorm1d(hidden_size // 4)  # BatchNorm for hidden_size // 4
        self.fc4 = nn.Linear(hidden_size // 4, output_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.batch_norm1(x)
        x = F.relu(self.fc2(x))
        x = self.batch_norm2(x)
        x = F.relu(self.fc3(x))
        x = self.batch_norm3(x)
        x = self.fc4(x)
        return x
    
# Enhanced Residual Block with Two Linear Layers
class EnhancedResidualBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(EnhancedResidualBlock, self).__init__()
        # First linear layer
        self.fc1 = nn.Linear(in_features, out_features)
        self.batch_norm1 = nn.BatchNorm1d(out_features)
        self.dropout1 = nn.Dropout(0.3)
        
        # Second linear layer
        self.fc2 = nn.Linear(out_features, out_features)
        self.batch_norm2 = nn.BatchNorm1d(out_features)
        self.dropout2 = nn.Dropout(0.3)
        
        # Shortcut connection to match dimensions if needed
        self.shortcut = nn.Sequential()
        if in_features != out_features:
            self.shortcut = nn.Linear(in_features, out_features)
        
    def forward(self, x):
        residual = x
        
        # First transformation
        x = F.relu(self.fc1(x))
        x = self.batch_norm1(x)
        x = self.dropout1(x)
        
        # Second transformation
        x = F.relu(self.fc2(x))
        x = self.batch_norm2(x)
        x = self.dropout2(x)
        
        # Add residual connection and apply final activation
        x += self.shortcut(residual)
        x = F.relu(x)
        return x

# Optimized Neural Network
class OptimizedNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1):
        super(OptimizedNeuralNetwork, self).__init__()
        # Initial layer
        self.initial = nn.Linear(input_size, hidden_size)
        
        # Stack of enhanced residual blocks
        self.residual_block1 = EnhancedResidualBlock(hidden_size, hidden_size)
        self.residual_block2 = EnhancedResidualBlock(hidden_size, hidden_size)
        self.residual_block3 = EnhancedResidualBlock(hidden_size, hidden_size // 2)
        self.residual_block4 = EnhancedResidualBlock(hidden_size // 2, hidden_size // 4)
        
        # Final output layer
        self.final = nn.Linear(hidden_size // 4, output_size)
        
    def forward(self, x):
        x = F.relu(self.initial(x))
        
        # Pass through residual blocks
        x = self.residual_block1(x)
        x = self.residual_block2(x)
        x = self.residual_block3(x)
        x = self.residual_block4(x)
        
        x = self.final(x)
        return x
    
class GNN(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=1):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim // 2)
        self.fc = nn.Linear(hidden_dim // 2, output_dim)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = x.mean(dim=0)  # Global pooling (e.g., mean over nodes)
        return self.fc(x)