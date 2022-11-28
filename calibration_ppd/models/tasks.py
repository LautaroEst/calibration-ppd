
import json
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


class InitModelFromTokenizer(Task):

    def __init__(self,name,**args):
        self.name = name
        self.model_cls = getattr(tm,name)
        self.args = args
        
    def run(self,tokenizer):
        self.num_embeddings = len(tokenizer)
        self.pad_idx = tokenizer.pad_token_id
        model = self.model_cls(num_embeddings=self.num_embeddings,pad_idx=self.pad_idx,**self.args)
        return model

    def save(self,output,output_dir):
        with open(os.path.join(output_dir,"config.json"),"w") as f:
            json.dump({
                "name": self.name,
                "num_embeddings": self.num_embeddings,
                "pad_idx": self.pad_idx,
                **self.args
            },f)
        with open(os.path.join(output_dir,"state_dict.pkl"),"wb") as f:
            torch.save(output.state_dict(),f)

    def load(self,output_dir):
        with open(os.path.join(output_dir,"config.json"),"r") as f:
            config = json.load(f)
        name = config.pop("name")
        num_embeddings = config.pop("num_embeddings")
        pad_idx = config.pop("pad_idx")

        model_cls = getattr(tm,name)
        model = model_cls(num_embeddings=num_embeddings,pad_idx=pad_idx,**config)

        with open(os.path.join(output_dir,"state_dict.pkl"),"rb") as f:
            state_dict = torch.load(f)
        model.load_state_dict(state_dict)
        return model        
        
