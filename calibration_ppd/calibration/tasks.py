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


class DiscriminativeModelCalibration(Task):

    _losses = {
        "affine_log": AffineCalLogLoss,
        "affine_ece": AffineCalECE,
        "affine_log_plus_ece": AffineCalLogLossPlusECE,
        "affine_brier": AffineCalBrier,
    }

    def __init__(self,priors,bias,calibration_loss):
        self.priors = priors 
        self.bias = bias
        self.calibration_loss = self._losses[calibration_loss]
        
    def run(self,model_outputs):
        logsigmoid = torch.nn.LogSigmoid()
        targetsTraining = torch.tensor(model_outputs["calibration_train"]["labels"],dtype=torch.int64)
        logPosteriorsTraining = logsigmoid(torch.tensor(model_outputs["calibration_train"]["logits"],dtype=torch.float32))
        targetsValidation = torch.tensor(model_outputs["calibration_validation"]["labels"],dtype=torch.int64)
        logPosteriorsValidation = logsigmoid(torch.tensor(model_outputs["calibration_validation"]["logits"],dtype=torch.float32))

        calibratedLogPosteriors, parameters = calibrate(
            logPosteriorsTraining, 
            targetsTraining, 
            logPosteriorsValidation, 
            self.calibration_loss, 
            bias=self.bias, 
            priors=self.priors, 
            quiet=True
        )
        
        return {
            "logpostiriors": calibratedLogPosteriors.cpu().detach().numpy(),
            "labels": targetsValidation.cpu().detach().numpy(),
            "parameters": parameters
        }

    def save(self,output,output_dir):
        # np.savez_compressed(output_dir+"/results.npz",logits=output["logits"])
        with open(f"{output_dir}/results.pkl","wb") as f:
            pickle.dump(output,f)
        

    def load(self,output_dir):
        with open(output_dir+"_results.pkl", 'rb') as f:
            data = pickle.load(f)
        return data



