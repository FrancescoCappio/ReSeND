import numpy as np
import math
import torch
from torch import nn
from tqdm import tqdm

class LogUnbuffered:

    def __init__(self, args, stream, file):
        self.args = args
        self.stream = stream
        self.file = file

    def write(self, data):
        if self.args.distributed and self.args.global_rank > 0 and not self.args.debug:
            return
        self.stream.write(data)
        self.file.write(data)    # Write the data of stdout here to a text file as well
        self.flush()
    
    def flush(self):
        self.stream.flush()
        self.file.flush()

def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        total_params+=param
    return total_params

