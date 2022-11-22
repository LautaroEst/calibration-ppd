

from ..core import Task
from .metrics import METRICS
from .loss_functions import load_loss_function
import torch


def load_metric(name,**args):

    if name not in METRICS.keys():
        raise ValueError(f"metric {name} not supported")
    
    metric_cls = METRICS[name]
    metric = metric_cls(**args)
    return metric


class LoadMetrics(Task):
    
    def __init__(self,**metrics):
        self.metrics = metrics

    def run(self):
        metrics = {name: load_metric(name,**args) for name, args in self.metrics.items()}
        return metrics


class LoadLossFunction(Task):
    
    def __init__(self,name,**args):
        self.name = name
        self.args = args

    def run(self):
        return load_loss_function(self.name,**self.args)

    
class EvaluateModel(Task):

    def __init__(self):
        pass

    def evaluate(self,model,sample):
        labels = sample.pop("label")
        outputs = model(**sample)
        logits = outputs["logits"]
        

    def run(self,model,dataset):
        model.eval()
        with torch.no_grad():
            dataset.map(lambda sample: self.evaluate(model,sample))


