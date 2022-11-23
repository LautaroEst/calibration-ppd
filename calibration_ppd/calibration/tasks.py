from .calibration import calibrate, AffineCalLogLoss, AffineCalBrier, AffineCalECE, AffineCalLogLossPlusECE
from ..core import Task
import numpy as np
import torch
import pickle
from scipy.special import expit


class LoadModelPredictions(Task):

    def __init__(self,path):
        self.path = path

    def run(self):
        with open(self.path, "rb") as f:
            results = pickle.load(f)
        return results


class LogisticRegression(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.temp = torch.nn.Parameter(torch.tensor(1e-3))
        self.bias = torch.nn.Parameter(torch.tensor(0.))

    def forward(self,input):
        return self.temp * input + self.bias


def train_calibrator(train_logits,train_labels,prior_class1,epochs,lr,device):

    model = LogisticRegression()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)

    device = torch.device(device)

    N = torch.tensor([(train_labels == 0).sum(), (train_labels == 1).sum()])
    p = torch.tensor([(1-prior_class1), prior_class1])
    w = p[train_labels]/N[train_labels]
    criterion = torch.nn.BCEWithLogitsLoss(weight=w.to(device))

    train_logits = torch.from_numpy(train_logits).to(device,dtype=torch.float)
    train_labels = torch.from_numpy(train_labels).to(device,dtype=torch.float)
    model.to(device)
    for e in range(epochs):

        optimizer.zero_grad()

        logits = model(train_logits)
        loss = criterion(logits,train_labels)
        loss.backward()
        optimizer.step()

    return model, {"temp": model.temp.cpu().detach().numpy(), "bias": model.bias.cpu().detach().numpy()}

def calibrate_logits(model,criterion,logits,labels):

    model.eval()
    device = model.device
    with torch.no_grad():
        logits = torch.from_numpy(logits).to(device)
        calibrated_logits = model(logits)

    return calibrated_logits


class DiscriminativeModelCalibration(Task):

    _losses = {
        "affine_log": AffineCalLogLoss,
        "affine_ece": AffineCalECE,
        "affine_log_plus_ece": AffineCalLogLossPlusECE,
        "affine_brier": AffineCalBrier,
    }

    def __init__(self,prior_class1,bias,calibration_loss,epochs,lr,device,run_on_test=False):
        self.prior_class1 = prior_class1 
        self.bias = bias
        self.calibration_loss = self._losses[calibration_loss]
        self.epochs = epochs
        self.lr = lr
        self.device = device
        self.run_on_test = run_on_test
        
    def run(self,model_outputs):
        labels_training = model_outputs["calibration_train"]["labels"]
        logits_training = model_outputs["calibration_train"]["logits"]

        # calibrated_logits, parameters = calibrate(
        #     logits_training, 
        #     labels_training, 
        #     logits_validation, 
        #     self.calibration_loss, 
        #     bias=self.bias, 
        #     priors=self.priors, 
        #     quiet=True
        # )

        model, parameters = train_calibrator(
            logits_training,
            labels_training,
            prior_class1=self.prior_class1,
            epochs=self.epochs,
            lr=self.lr,
            device=torch.device(self.device)
        )
        criterion = torch.nn.BCEWithLogitsLoss()

        logits_validation = model_outputs["calibration_validation"]["logits"]
        labels_validation = model_outputs["calibration_validation"]["labels"]
        cal_logits_val = calibrate_logits(model,logits_validation,labels_validation)
        cal_logloss_val = criterion(cal_logits_val,labels_validation).item()
        uncal_logloss_val = criterion(logits_validation,labels_validation).item()

        if self.run_on_test:
            logits_test = model_outputs["training_validation"]["logits"]
            labels_test = model_outputs["training_validation"]["labels"]
            cal_logits_test = calibrate_logits(model,logits_test,labels_test)
            cal_logloss_test = criterion(cal_logits_test,labels_test).item()
            uncal_logloss_test = criterion(logits_test,labels_test).item()
            cal_logits_test = cal_logits_test.cpu().detach().numpy()
            logits_test = logits_test.cpu().detach().numpy()
        else:
            logits_test = None
            cal_logits_test = None
            cal_logloss_test, uncal_logloss_test = "N/A", "N/A"

        return {
            "logits": {
                "calibrated_validation": cal_logits_val.cpu().detach().numpy(),
                "uncalibrated_validation": logits_validation.cpu().detach().numpy(),
                "calibrated_test": cal_logits_test,
                "uncalibrated_test": logits_test,
            },
            "logloss": {
                "calibrated_validation": cal_logloss_val,
                "uncalibrated_validation": uncal_logloss_val,
                "calibrated_test": cal_logloss_test,
                "uncalibrated_test": uncal_logloss_test
            },
            "parameters": parameters,
            "prior_class1": self.prior_class1
        }

    def save(self,output,output_dir):
        # np.savez_compressed(output_dir+"/results.npz",logits=output["logits"])
        with open(f"{output_dir}/results.pkl","wb") as f:
            pickle.dump(output,f)
        

    def load(self,output_dir):
        with open(output_dir+"_results.pkl", 'rb') as f:
            data = pickle.load(f)
        return data



