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
    w = p[train_labels]/N[train_labels]#*len(train_labels)
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

def calibrate_logits(model,logits):

    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        logits = logits.to(device)
        calibrated_logits = model(logits)

    return calibrated_logits

def calculate_logloss(model, logits, labels, prior):
    device = next(model.parameters()).device

    N = torch.tensor([(labels == 0).sum(), (labels == 1).sum()],device=device)
    p = torch.tensor([(1-prior), prior],device=device)
    w = p[labels.type(torch.long)]/N[labels.type(torch.long)]#*len(labels)
    
    criterion = torch.nn.BCEWithLogitsLoss(weight=w.to(device))
    # criterion = torch.nn.BCEWithLogitsLoss()
    cal_logits = calibrate_logits(model,logits)
    cal_logloss = criterion(cal_logits,labels).item()
    uncal_logloss = criterion(logits,labels).item()
    cal_logits = cal_logits.cpu().detach().numpy()
    uncal_logits = logits.cpu().detach().numpy()
    return uncal_logits, cal_logits, cal_logloss, uncal_logloss


class DiscriminativeModelCalibration(Task):

    _losses = {
        "affine_log": AffineCalLogLoss,
        "affine_ece": AffineCalECE,
        "affine_log_plus_ece": AffineCalLogLossPlusECE,
        "affine_brier": AffineCalBrier,
    }

    def __init__(self, training_prior, evaluation_prior, calibration_loss, epochs,lr, device, run_on_test=False):
        self.training_prior = training_prior
        self.evaluation_prior = evaluation_prior
        self.bias = True
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
            prior_class1=self.training_prior,
            epochs=self.epochs,
            lr=self.lr,
            device=torch.device(self.device)
        )
        # criterion = torch.nn.BCEWithLogitsLoss()

        logits_validation = torch.tensor(model_outputs["calibration_validation"]["logits"],dtype=torch.float,device=torch.device(self.device))
        labels_validation = torch.tensor(model_outputs["calibration_validation"]["labels"],dtype=torch.float,device=torch.device(self.device))
        uncal_logits_validation, cal_logits_validation, cal_logloss_validation, uncal_logloss_validation = calculate_logloss(
            model, 
            logits_validation, 
            labels_validation, 
            self.evaluation_prior
        )

        # cal_logits_val = calibrate_logits(model,logits_validation)
        # cal_logloss_val = criterion(cal_logits_val,labels_validation).item()
        # uncal_logloss_val = criterion(logits_validation,labels_validation).item()

        if self.run_on_test:
            logits_test = torch.tensor(model_outputs["training_validation"]["logits"],dtype=torch.float,device=torch.device(self.device))
            labels_test = torch.tensor(model_outputs["training_validation"]["labels"],dtype=torch.float,device=torch.device(self.device))
            # cal_logits_test = calibrate_logits(model,logits_test)
            # cal_logloss_test = criterion(cal_logits_test,labels_test).item()
            # uncal_logloss_test = criterion(logits_test,labels_test).item()
            # cal_logits_test = cal_logits_test.cpu().detach().numpy()

            uncal_logits_test, cal_logits_test, cal_logloss_test, uncal_logloss_test = calculate_logloss(
                model, 
                logits_test, 
                labels_test, 
                self.evaluation_prior
            )

            logits_test = logits_test.cpu().detach().numpy()
            labels_test = labels_test.cpu().detach().numpy()
        else:
            logits_test = None
            labels_test = None
            cal_logits_test = None
            cal_logloss_test, uncal_logloss_test = "N/A", "N/A"

        return {
            "logits": {
                "calibrated_validation": cal_logits_validation,
                "uncalibrated_validation": uncal_logits_validation,
                "calibrated_test": cal_logits_test,
                "uncalibrated_test": uncal_logits_test,
            },
            "labels": {
                "validation": labels_validation.cpu().detach().numpy(),
                "test": labels_test
            },
            "logloss": {
                "calibrated_validation": cal_logloss_validation,
                "uncalibrated_validation": uncal_logloss_validation,
                "calibrated_test": cal_logloss_test,
                "uncalibrated_test": uncal_logloss_test
            },
            "parameters": parameters,
            "training_prior": self.training_prior,
            "evaluation_prior": self.evaluation_prior
        }

    def save(self,output,output_dir):
        # np.savez_compressed(output_dir+"/results.npz",logits=output["logits"])
        with open(f"{output_dir}/results.pkl","wb") as f:
            pickle.dump(output,f)
        

    def load(self,output_dir):
        with open(output_dir+"_results.pkl", 'rb') as f:
            data = pickle.load(f)
        return data



