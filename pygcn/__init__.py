from __future__ import print_function
from __future__ import division

from .layers import *
from .models import *
from .utils import *
import random

# def setup_seed(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.backends.cudnn.deterministic = True
#
#
# setup_seed(10086)