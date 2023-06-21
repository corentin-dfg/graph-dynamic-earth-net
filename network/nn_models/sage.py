import torch
import torch_geometric as pyg

class SAGE(torch.nn.Module):
    """
    Inductive Representation Learning on Large Graphs
    (https://proceedings.neurips.cc/paper_files/paper/2017/hash/5dd9db5e033da9c6fb5ba83c7a7ebea9-Abstract.html)
    """
    
    def __init__(self, input_dim, output_dim, widths, norm="batch", last_relu=True):
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
                pyg.nn.SAGEConv(
                    in_channels=widths[i],
                    out_channels=widths[i + 1]
                )
            )
            if nl is not None:
                self.layers.append(nl(widths[i + 1]))

            if last_relu:
                self.layers.append(torch.nn.ReLU(inplace=True))
            elif i < self.n_stages - 1:
                self.layers.append(torch.nn.ReLU(inplace=True))
        
        self.layers.append(
            pyg.nn.Linear(
                in_channels=widths[-2],
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