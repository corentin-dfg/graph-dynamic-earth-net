import torch

class MLP(torch.nn.Module):
    def __init__(self, input_dim, output_dim, widths, norm="batch", last_relu=True):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_stages = len(widths)+1
        widths = [input_dim]+widths+[output_dim]

        layers = []
        if norm == "batch":
            nl = torch.nn.BatchNorm1d
        elif norm == "instance":
            nl = lambda num_feats: torch.nn.InstanceNorm1d(num_features=num_feats, affine=True)
        elif norm == "group":
            nl = lambda num_feats: torch.nn.GroupNorm(num_channels=num_feats, num_groups=4)
        else:
            nl = None

        for i in range(self.n_stages - 1):
            layers.append(
                torch.nn.Linear(in_features=widths[i], out_features=widths[i+1])
            )
            if nl is not None:
                layers.append(nl(widths[i + 1]))

            if last_relu:
                layers.append(torch.nn.ReLU(inplace=True))
            elif i < self.n_stages - 1:
                layers.append(torch.nn.ReLU(inplace=True))
        
        layers.append(
            torch.nn.Linear(in_features=widths[-2], out_features=widths[-1])
        )

        self.model = torch.nn.Sequential(*layers)

    def forward(self, data):
        return self.model(data["region"].x)