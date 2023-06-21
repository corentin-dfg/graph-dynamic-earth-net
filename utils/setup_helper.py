import random
import numpy as np
import psutil
import torch

# Check array/tensor size
mem_size_of = lambda a: a.element_size() * a.nelement()
gb = lambda bs: bs / 2. ** 30


def get_sys_mem():
    p = psutil.Process()
    pmem = p.memory_info()
    return gb(pmem.rss), gb(pmem.vms)


def load_weights(weights_dir, device, key='state_dict'):
    map_location = lambda storage, loc: storage.cuda(device.index) if torch.cuda.is_available() else storage
    weights_dict = None
    if weights_dir is not None:
        weights_dict = torch.load(weights_dir, map_location=map_location)
    return weights_dict

def make_deterministic(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Important also

