
import torch
from torch import nn, optim

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_only


class TBLogger(TensorBoardLogger):
    @rank_zero_only
    def log_metrics(self, metrics, step):
        metrics.pop('epoch', None)
        return super().log_metrics(metrics, step)


class ClassificationLightningModule(pl.LightningModule):

    def __init__(self,model,loss_fn,optimizer,metrics):
        super().__init__()
        self.model = model
        if isinstance(loss_fn,torch.nn.BCEWithLogitsLoss):
            self.loss_fn = lambda input, target: loss_fn(input.squeeze(dim=1),target.float())
        else:
            self.loss_fn = loss_fn
        self._optimizer = optimizer
        self.metrics = metrics

    def forward(self,**inputs):
        outputs = self.model(**inputs)
        return outputs

    @staticmethod
    def create_log_dict(loss,labels,logits,metrics,split="train"):
        log_dict = {f"loss/{split}": loss}
        for metric_name, metric in metrics.items():
            log_dict[f"{metric_name}/{split}"] = metric.compute(reference=labels,logits=logits)
        return log_dict

    def training_step(self,batch):
        labels = batch.pop("label")
        outputs = self(**batch)
        logits = outputs["logits"]
        loss = self.loss_fn(logits,labels)
        log_dict = self.create_log_dict(
            loss.item(),
            labels.cpu().detach().numpy(),
            logits.cpu().detach().numpy(),
            self.metrics,
            split="train"
        )
        self.log_dict(log_dict,logger=True)
        return loss

    def validation_step(self,batch,batch_idx):
        labels = batch.pop("label")
        outputs = self(**batch)
        logits = outputs["logits"]
        loss = self.loss_fn(logits,labels)
        return {
            "num_samples": len(labels),
            "loss": loss.item(),
            "logits": logits,
            "labels": labels
        }

    def validation_epoch_end(self,outputs):
        num_samples = sum([output["num_samples"] for output in outputs])
        avg_loss = sum([output["loss"] for output in outputs]) / num_samples
        all_logits = torch.vstack([output["logits"] for output in outputs]).cpu().detach().numpy()
        all_labels = torch.hstack([output["labels"] for output in outputs]).cpu().detach().numpy()
        log_dict = self.create_log_dict(avg_loss,all_labels,all_logits,self.metrics,split="validation")
        self.log_dict(log_dict,logger=True)

    def backward(self,loss,optimizer,optimizer_idx):
        loss.backward()

    def configure_optimizers(self):
        return self._optimizer

    def optimizer_step(
        self,
        epoch=None,
        batch_idx=None,
        optimizer=None,
        optimizer_idx=None,
        optimizer_closure=None,
        on_tpu=None,
        using_native_amp=None,
        using_lbfgs=None
    ):
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()






