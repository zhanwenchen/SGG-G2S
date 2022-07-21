from random import seed as random_seed
from numpy.random import seed as np_random_seed
from torch import manual_seed as torch_manual_seed
from torch.backends import cudnn
from torch.cuda import manual_seed_all


def setup_seed(seed):
    torch_manual_seed(seed)
    manual_seed_all(seed)
    np_random_seed(seed)
    random_seed(seed)
    cudnn.deterministic = True
