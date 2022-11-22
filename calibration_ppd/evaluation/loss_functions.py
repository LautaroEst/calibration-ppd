
import torch

torch_losses = torch.nn.modules.loss.__all__

def load_loss_function(name,**args):
    
    if name in torch_losses:
        loss = getattr(torch.nn,name)
        return loss(**args)

    raise ValueError(f"Loss function {name} not supported")
