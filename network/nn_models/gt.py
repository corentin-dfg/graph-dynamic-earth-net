"""
Code of GraphTransformer adapted from repo:
https://github.com/vijaydwivedi75/lrgb/blob/main/graphgps/layer/san_layer.py
"""

import torch
import torch.nn.functional as F
import torch_geometric as pyg
import numpy as np
from torch_scatter import scatter

class InputEncoder(torch.nn.Module):
    """Laplace Positional Embedding node encoder.
    LapPE of size dim_pe will get appended to each node feature vector.
    If `expand_x` set True, original node features will be first linearly
    projected to (dim_emb - dim_pe) size and the concatenated with LapPE.
    Args:
        dim_emb: Size of final node embedding
        expand_x: Expand node features `x` from dim_in to (dim_emb - dim_pe)
        dim_pe: Size of Laplace PE embedding
        model_type: Encoder NN model type for PEs [Transformer, DeepSet]
        n_layers: Num. layers in PE encoder model
        n_heads: Num. attention heads in Trf PE encoder
        max_freqs: Num. eigenvectors (frequencies)
        norm_type: Raw PE normalization layer type
    """

    def __init__(self, dim_in, dim_emb, dim_pe=8, max_freqs=10, norm_type='none'):
        super().__init__()

        # Initial projection of eigenvalue and the node's eigenvector value
        self.linear_A = torch.nn.Linear(dim_in, dim_emb)
        self.linear_C = torch.nn.Linear(dim_pe, dim_emb)
        
    def forward(self, batch):
        if not (hasattr(batch, 'EigVals') and hasattr(batch, 'EigVecs')):
            raise ValueError("Precomputed eigen values and vectors are "
                             f"required for {self.__class__.__name__}; "
                             "set config 'posenc_LapPE.enable' to True")
        
        # Concatenate final PEs to input embedding
        batch["region"].x = self.linear_A(batch["region"].x) + self.linear_C(batch.EigVecs)

        return batch

class MultiHeadAttentionLayer(torch.nn.Module):
    """Multi-Head Graph Attention Layer.
    Ported to PyG from original repo:
    https://github.com/DevinKreuzer/SAN/blob/main/layers/graph_transformer_layer.py
    """

    def __init__(self, gamma, in_dim, out_dim, num_heads, use_bias):
        super().__init__()

        in_dim_edges = 1
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.gamma = gamma

        self.Q = torch.nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.K = torch.nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.E = torch.nn.Linear(in_dim_edges, out_dim * num_heads, bias=use_bias)

        self.V = torch.nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)

    def propagate_attention(self, batch):
        src = batch.K_h[batch.edge_index[0]]  # (num real edges) x num_heads x out_dim
        dest = batch.Q_h[batch.edge_index[1]]  # (num real edges) x num_heads x out_dim
        score = torch.mul(src, dest)  # element-wise multiplication

        # Scale scores by sqrt(d)
        score = score / np.sqrt(self.out_dim)

        # Use available edge features to modify the scores for edges
        score = torch.mul(score, batch.E)  # (num real edges) x num_heads x out_dim
        score = torch.exp(score.sum(-1, keepdim=True).clamp(-5, 5))  # (num real edges) x num_heads x 1

        # Apply attention score to each source node to create edge messages
        msg = batch.V_h[batch.edge_index[0]] * score  # (num real edges) x num_heads x out_dim
        # Add-up real msgs in destination nodes as given by batch.edge_index[1]
        batch.wV = torch.zeros_like(batch.V_h)  # (num nodes in batch) x num_heads x out_dim
        scatter(msg, batch.edge_index[1], dim=0, out=batch.wV, reduce='add')

        # Compute attention normalization coefficient
        batch.Z = score.new_zeros(batch.size(0), self.num_heads, 1)  # (num nodes in batch) x num_heads x 1
        scatter(score, batch.edge_index[1], dim=0, out=batch.Z, reduce='add')

    def forward(self, batch):
        Q_h = self.Q(batch["region"].x)
        K_h = self.K(batch["region"].x)
        E = self.E(batch.edge_attr)

        V_h = self.V(batch["region"].x)

        # Reshaping into [num_nodes, num_heads, feat_dim] to
        # get projections for multi-head attention
        batch.Q_h = Q_h.view(-1, self.num_heads, self.out_dim)
        batch.K_h = K_h.view(-1, self.num_heads, self.out_dim)
        batch.E = E.view(-1, self.num_heads, self.out_dim)

        batch.V_h = V_h.view(-1, self.num_heads, self.out_dim)

        self.propagate_attention(batch)

        h_out = batch.wV / (batch.Z + 1e-6)

        return h_out


class GTLayer(torch.nn.Module):
    """GraphTransformerLayer from SAN.
    Ported to PyG from original repo:
    https://github.com/DevinKreuzer/SAN/blob/main/layers/graph_transformer_layer.py
    """

    def __init__(self, gamma, in_dim, out_dim, num_heads, dropout=0.0, norm="batch",
                 residual=True, use_bias=False):
        super().__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual

        if norm == "batch":
            self.nl = torch.nn.BatchNorm1d
        elif norm == "instance":
            self.nl = torch.nn.InstanceNorm1d(affine=True)
        elif norm == "group":
            self.nl = lambda num_feats: torch.nn.GroupNorm(
                num_channels=num_feats,
                num_groups=4,
                affine=True
            )
        else:
            self.nl = None

        self.attention = MultiHeadAttentionLayer(gamma=gamma,
                                                 in_dim=in_dim,
                                                 out_dim=out_dim // num_heads,
                                                 num_heads=num_heads,
                                                 use_bias=use_bias)

        self.O_h = torch.nn.Linear(out_dim, out_dim)

        if self.nl is not None:
            self.norm1_h = self.nl(out_dim)

        # FFN for h
        self.FFN_h_layer1 = torch.nn.Linear(out_dim, out_dim * 2)
        self.FFN_h_layer2 = torch.nn.Linear(out_dim * 2, out_dim)

        if self.nl is not None:
            self.norm2_h = self.nl(out_dim)

    def forward(self, batch):
        h = batch["region"].x
        h_in1 = h  # for first residual connection

        # multi-head attention out
        h_attn_out = self.attention(batch)

        # Concat multi-head outputs
        h = h_attn_out.view(-1, self.out_channels)

        h = F.dropout(h, self.dropout, training=self.training)

        h = self.O_h(h)

        if self.residual:
            h = h_in1 + h  # residual connection

        if self.nl is not None:
            h = self.norm1_h(h)

        h_in2 = h  # for second residual connection

        # FFN for h
        h = self.FFN_h_layer1(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_h_layer2(h)

        if self.residual:
            h = h_in2 + h  # residual connection

        if self.nl is not None:
            h = self.norm2_h(h)

        batch["region"].x = h
        return batch
    
class GNNInductiveNodeHead(torch.nn.Module):
    """
    GNN prediction head for inductive node prediction tasks.
    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension. For binary prediction, dim_out=1.
    """

    def __init__(self, dim_in, dim_out, n_layers=3):
        super(GNNInductiveNodeHead, self).__init__()
        layers = []
        for l in range(n_layers-1):
            layers.append((pyg.nn.Linear(dim_in,dim_in,bias=True),'x -> x'))
        layers.append((pyg.nn.Linear(dim_in,dim_out,bias=True),'x -> x'))
        self.layer_post_mp = pyg.nn.Sequential('x',layers)

    def forward(self, data):
        return self.layer_post_mp(data["region"].x)

class GraphTransformer(torch.nn.Module):
    """
    A Generalization of Transformer Networks to Graphs
    (https://arxiv.org/abs/2012.09699)
    """
    def __init__(self, input_dim, output_dim, widths, norm='batch', num_heads=4, dim_edge_attr=1, gamma=1e-1):
        super().__init__()
        self.dim_edge_attr = dim_edge_attr
        self.n_stages = len(widths)+1
        widths = [widths[0]] + widths

        self.encoder = InputEncoder(input_dim, widths[0], dim_pe=10)

        self.layers = torch.nn.ModuleList()
        for i in range(self.n_stages - 1):
            self.layers.append(GTLayer(gamma=gamma,
                                in_dim=widths[i],
                                out_dim=widths[i+1],
                                num_heads=num_heads,
                                dropout=0.0,
                                norm=norm,
                                residual=True))

        self.post_mp = GNNInductiveNodeHead(dim_in=widths[-1], dim_out=output_dim, n_layers=1)

    def forward(self, data):
        data = self.encoder(data)

        batch_split = data._slice_dict["region"]['node_index']
        data.batch = torch.arange(0,len(batch_split)-1).repeat_interleave(batch_split[1:]-batch_split[:-1]).to(data["region"].x.device)

        data.edge_index = data["region","spatial","region"].edge_index
        data.edge_attr = torch.ones((data.edge_index.shape[1], self.dim_edge_attr), device=data["region"].x.device)

        data = self.layers[0](data)
        data = self.layers[1](data)

        data.edge_index = data["region","temporal","region"].edge_index
        data.edge_attr = torch.ones((data.edge_index.shape[1], self.dim_edge_attr), device=data["region"].x.device)

        data = self.layers[2](data)
        data = self.layers[3](data)

        return self.post_mp(data)