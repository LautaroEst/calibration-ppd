from .calibration import calibrate,AffineCalLogLoss
from ..core import Task
import numpy as np
import torch
import pickle
from scipy.special import expit



class DiscriminativeModelCalibration(Task):

    def __init__(self,model):
        logsigmoid = torch.nn.LogSigmoid()
        modelOutputs = self.load("00_train_system/"+model)        
        self.model = model
        self.priors = None #podemos probar cambiando esto
        self.targetsTraining = torch.tensor(modelOutputs["train"]["labels"],dtype=torch.int64)
        self.logPosteriorsTraining = logsigmoid(torch.tensor(modelOutputs["train"]["logits"],dtype=torch.float32))
        self.logPosteriorsValidation = logsigmoid(torch.tensor(modelOutputs["validation"]["logits"],dtype=torch.float32))

    def run(self):
        calibratedLogPosteriors, parameters = calibrate(self.logPosteriorsTraining, self.targetsTraining, self.logPosteriorsValidation, AffineCalLogLoss, bias=True, priors=self.priors, quiet=True)
        #cambiando el parámetro bias habilita o deshabilita el temp_scaling
        return {"logprobs":calibratedLogPosteriors.detach().numpy()}

    def save(self,output,output_dir):
        np.savez_compressed(output_dir+"/results.npz",logprobs=output["logprobs"])

    def load(self,output_dir):
        with open(output_dir+"_results.pkl", 'rb') as f:
            data = pickle.load(f)
        return data



