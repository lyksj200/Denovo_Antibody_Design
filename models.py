# src/models.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

# Import model parameters from our config file
from src.config import NODE_FEATURE_DIM, HIDDEN_DIM, NUM_GNN_LAYERS, DROPOUT_RATE


class EpitopeGNN(nn.Module):
    """
    Graph Attention Network (GAT) for predicting protein antigen epitopes.
    """
    def __init__(self,
                 input_dim=NODE_FEATURE_DIM,
                 hidden_dim=HIDDEN_DIM,
                 output_dim=1,
                 num_layers=NUM_GNN_LAYERS,
                 dropout=DROPOUT_RATE,
                 heads=4):
        """
        Initializes the layers of the GNN.

        Args:
            input_dim (int): Dimensionality of the input node features.
            hidden_dim (int): Dimensionality of the hidden layers.
            output_dim (int): Dimensionality of the output (1 for binary classification).
            num_layers (int): The number of GAT layers.
            dropout (float): Dropout probability.
            heads (int): Number of attention heads in each GAT layer.
        """
        super(EpitopeGNN, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        # --- Input Layer ---
        # A linear layer to project the initial node features into the hidden dimension
        self.input_lin = nn.Linear(input_dim, hidden_dim)

        # --- GNN Layers ---
        # A list to hold the GAT layers
        self.convs = nn.ModuleList()
        # A list for normalization layers, crucial for stable training
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            # We use GATv2Conv which is an improved version of the original GAT
            # It takes hidden_dim as input and produces hidden_dim as output
            # (heads * hidden_dim / heads = hidden_dim)
            conv = GATv2Conv(hidden_dim, hidden_dim, heads=heads, concat=True, dropout=dropout)
            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(hidden_dim * heads))

        # --- Output Layer ---
        # A final linear layer to map the node embeddings to a single prediction score (logit)
        self.output_lin = nn.Linear(hidden_dim * heads, output_dim)


    def forward(self, data):
        """
        Defines the forward pass of the model.

        Args:
            data (torch_geometric.data.Batch): A batch of graph data from the DataLoader.

        Returns:
            torch.Tensor: The output logits for each node in the batch. Shape: [num_nodes, 1]
        """
        # Extract node features and graph connectivity from the batch object
        x, edge_index = data.x, data.edge_index

        # 1. Apply input layer and initial activation
        x = self.input_lin(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 2. Go through the GAT layers with skip connections
        for i in range(self.num_layers):
            # Apply GAT layer
            x = self.convs[i](x, edge_index)
            # Apply Layer Normalization
            x = self.norms[i](x)
            # Apply activation function
            x = F.relu(x)
            # Apply dropout
            x = F.dropout(x, p=self.dropout, training=self.training)

        # 3. Apply final output layer to get logits
        x = self.output_lin(x)

        return x


if __name__ == '__main__':
    # This is for testing purposes to see if the model can be created
    # and can process a dummy batch of data.
    from torch_geometric.data import Data, Batch

    print("Testing model instantiation...")
    model = EpitopeGNN()
    print(model)

    # Create a dummy graph
    num_nodes = 10
    num_edges = 20
    dummy_x = torch.randn(num_nodes, NODE_FEATURE_DIM)
    dummy_edge_index = torch.randint(0, num_nodes, (2, num_edges), dtype=torch.long)
    dummy_data = Data(x=dummy_x, edge_index=dummy_edge_index)
    
    # Create a dummy batch from a list of graphs
    dummy_batch = Batch.from_data_list([dummy_data, dummy_data])
    
    print("\nTesting model forward pass...")
    try:
        output = model(dummy_batch)
        print(f"Forward pass successful!")
        print(f"Input batch had {dummy_batch.num_nodes} nodes.")
        print(f"Output tensor shape: {output.shape}")
        assert output.shape == (dummy_batch.num_nodes, 1)
        print("Output shape is correct.")
    except Exception as e:
        print(f"An error occurred during the forward pass: {e}")