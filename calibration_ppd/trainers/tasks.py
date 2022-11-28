import os
from typing import Callable, Optional
from ..core import Task
from .pytorch import ClassificationLightningModule, TBLogger
import pytorch_lightning as pl
from torch import optim
import torch


class AdamWWithGradientClip(optim.AdamW):

    def __init__(self,params,lr,weight_decay,gradient_clip):
        super().__init__(params,lr=lr,weight_decay=weight_decay)
        self.gradient_clip_value = gradient_clip
        self._model_params = params

    def step(self, closure: Optional[Callable[[], float]] = ...) -> Optional[float]:
        torch.nn.utils.clip_grad_norm_(self._model_params, self.gradient_clip_value)
        return super().step(closure)

class InitAdamWOptimization(Task):

    def __init__(self,learning_rate,gradient_clip,weight_decay):
        self.learning_rate = learning_rate
        self.gradient_clip = gradient_clip
        self.weight_decay = weight_decay

    def run(self,model):
        if self.gradient_clip is None:
            optimizer = optim.AdamW(
                model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        else:
            optimizer = AdamWWithGradientClip(
                model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                gradient_clip=self.gradient_clip
            )
        
        return optimizer



class TrainSupervisedTorchModel(Task):

    def __init__(self,**train_args):
        self.train_args = train_args

    def run(self,model,loss_fn,train_data,validation_data,optimizer,metrics):
        output_dir = self.get_output_dir()
        
        pl_model = ClassificationLightningModule(model,loss_fn,optimizer,metrics)
        logger = TBLogger(output_dir,name="",version=0)
        trainer = pl.Trainer(
            default_root_dir=output_dir,
            logger=logger,
            **self.train_args
        )

        try:
            trainer.fit(pl_model,train_data,validation_data)
        except KeyboardInterrupt:
            ## TODO: implement saving
            pass

        return pl_model

    def save(self,output,output_dir):
        with open(os.path.join(output_dir,"state_dict.pkl"),"wb") as f:
            torch.save(output.model.state_dict(),f)

    def load(self,output_dir):
        ## TODO: implement loading
        raise NotImplementedError

    