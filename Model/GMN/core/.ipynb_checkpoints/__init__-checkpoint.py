# Ignore TF warnings due to numpy version 1.17.2
import warnings
warnings.simplefilter("ignore")
import os
os.environ["USE_TF"] = 'None' 
# os.environ["USE_TORCH"]= "True"

# use CPU
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

os.environ["TF_XLA_FLAGS"] = '--tf_xla_cpu_global_jit'


from .config import *
from .gnn_model import *
from .graph_factory_utils import *

