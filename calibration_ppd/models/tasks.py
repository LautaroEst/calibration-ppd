
from ..core import Task
import transformers
from . import torch_models as tm


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
        
