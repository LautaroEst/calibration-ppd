

from tqdm import tqdm
from ..core import Task
from .metrics import METRICS
from .loss_functions import load_loss_function
import torch
import pickle


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

    
class MakePredictions(Task):

    def __init__(self,device):
        self.device = device

    def run(self,model,**dataloaders):
        results = {}
        device = torch.device(self.device)
        model.to(device)
        model.eval()
        for split, dataloader in dataloaders.items():
            all_labels = []
            all_logits = []
            for batch in tqdm(dataloader):
                labels = batch.pop("label").view(-1)
                all_labels.append(labels)
                with torch.no_grad():
                    batch = {name: tensor.to(device) for name, tensor in batch.items()}
                    logits = model(**batch)["logits"].view(-1)
                    all_logits.append(logits)
            all_labels = torch.cat(all_labels).numpy()
            all_logits = torch.cat(all_logits).cpu().numpy()
            results[split] = {
                "labels": all_labels,
                "logits": all_logits
            }
        return results

    def save(self,output,output_dir):
        with open(f"{output_dir}/results.pkl","wb") as f:
            pickle.dump(output,f)


