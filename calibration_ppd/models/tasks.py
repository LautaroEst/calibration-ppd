
import os
import pickle
from ..core import Task
import transformers
from . import torch_models as tm
import torch
from collections import OrderedDict

class LoadPretrainedModel(Task):

    def __init__(self,model,source,**kwargs):
        self.model = getattr(transformers,model)
        self.source = source
        self.kwargs = kwargs

    def run(self):
        classifier = self.model.from_pretrained(self.source,**self.kwargs)
        return classifier

    def save(self,output,output_dir):
        output.save_pretrained(output_dir)

    def load(self,output_dir):
        classifier = self.model.from_pretrained(output_dir)
        return classifier


class InitTorchModel(Task):

    def __init__(self,name,**args):
        self.model_cls = getattr(tm,name)
        self.args = args
        
    def run(self,**input_args):
        all_args = {**input_args,**self.args}
        model = self.model_cls(**all_args)
        return model
        

# class LoadTorchModel(Task):

#     def __init__(self,path,task,**kwargs):
#         self.path = os.path.join(path,os.listdir(path)[0])

#     def run(self):
#         # with open(self.path,"rb") as f:
#         #     model = pickle.load(f)
#         state_dict = OrderedDict([(key.split("model.")[-1], params) for key, params in torch.load(self.path)["state_dict"].items()])
        
