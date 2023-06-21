import torch
import torch_geometric as pyg

class GAT(torch.nn.Module):
    """
    Graph Attention Networks
    (https://arxiv.org/abs/1710.10903)
    """
    def __init__(self, input_dim, output_dim, widths, norm="batch", num_heads=4, last_relu=True):
        super().__init__()

        self.n_stages = len(widths)+1
        widths = [input_dim]+widths+[output_dim]

        self.layers = torch.nn.ModuleList()
        if norm == "batch":
            nl = pyg.nn.norm.BatchNorm
        elif norm == "instance":
            nl = pyg.nn.norm.InstanceNorm
        elif norm == "group":
            nl = lambda num_feats: torch.nn.GroupNorm(
                num_channels=num_feats,
                num_groups=4,
                affine=True
            )
        else:
            nl = None

        for i in range(self.n_stages - 1):
            self.layers.append(
                pyg.nn.GATConv(
                    in_channels=widths[i] if i==0 else widths[i]*num_heads,
                    out_channels=widths[i + 1],
                    heads=num_heads,
                    concat=True
                )
            )
            if nl is not None:
                self.layers.append(nl(widths[i + 1]*num_heads))

            if last_relu:
                self.layers.append(torch.nn.ReLU(inplace=True))
            elif i < self.n_stages - 1:
                self.layers.append(torch.nn.ReLU(inplace=True))
        
        self.layers.append(
            pyg.nn.Linear(
                in_channels=widths[-2]*num_heads,
                out_channels=widths[-1]
            )
        )

    def forward(self, data):
        x, edge_index_spatial, edge_index_temporal = (data["region"].x, data["region", "spatial", "region"].edge_index, data["region", "temporal", "region"].edge_index)
        x = self.layers[0](x, edge_index_spatial)
        x = self.layers[1](x)
        x = self.layers[2](x)
        x = self.layers[3](x, edge_index_spatial)
        x = self.layers[4](x)
        x = self.layers[5](x)
        x = self.layers[6](x, edge_index_temporal)
        x = self.layers[7](x)
        x = self.layers[8](x)
        x = self.layers[9](x, edge_index_temporal)
        x = self.layers[10](x)
        x = self.layers[11](x)
        x = self.layers[12](x)
        return x