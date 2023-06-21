import torch
import warnings
from network.nn_models import gcn, sage, gat, resgatedgcn, mlp, gt
warnings.filterwarnings("ignore")

def init_net(net, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        print("Let's use", len(gpu_ids), "GPUs!")
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    return net

# Defines the network
def define_model(config, model, gpu_ids=[]):

    if model == 'mlp':
        net = mlp.MLP(
                input_dim=config['NETWORK']['INPUT_CHANNELS'],
                output_dim=config['NETWORK']['OUTPUT_CHANNELS'],
                widths=config['NETWORK']['WIDTHS'],
                norm=config['NETWORK']['NORM'])
    elif model == 'gcn':
        net = gcn.GCN(
                input_dim=config['NETWORK']['INPUT_CHANNELS'],
                output_dim=config['NETWORK']['OUTPUT_CHANNELS'],
                widths=config['NETWORK']['WIDTHS'],
                norm=config['NETWORK']['NORM'])
    elif model == 'sage':
        net = sage.SAGE(
                input_dim=config['NETWORK']['INPUT_CHANNELS'],
                output_dim=config['NETWORK']['OUTPUT_CHANNELS'],
                widths=config['NETWORK']['WIDTHS'],
                norm=config['NETWORK']['NORM'])
    elif model == 'gat':
        net = gat.GAT(
                input_dim=config['NETWORK']['INPUT_CHANNELS'],
                output_dim=config['NETWORK']['OUTPUT_CHANNELS'],
                widths=config['NETWORK']['WIDTHS'],
                norm=config['NETWORK']['NORM'],
                num_heads=config['NETWORK']['NUM_HEADS'])
    elif model == 'resgatedgcn':
        net = resgatedgcn.ResGatedGCN(
                input_dim=config['NETWORK']['INPUT_CHANNELS'],
                output_dim=config['NETWORK']['OUTPUT_CHANNELS'],
                widths=config['NETWORK']['WIDTHS'],
                norm=config['NETWORK']['NORM'])
    elif model == 'gt':
        net = gt.GraphTransformer(
                input_dim=config['NETWORK']['INPUT_CHANNELS'],
                output_dim=config['NETWORK']['OUTPUT_CHANNELS'],
                widths=config['NETWORK']['WIDTHS'],
                norm=config['NETWORK']['NORM'],
                num_heads=config['NETWORK']['NUM_HEADS'])
    else:
        raise NotImplementedError(f'The model name [{model}] is not recognized')
    return init_net(net, gpu_ids=gpu_ids)


