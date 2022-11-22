from calibration import calibrate,AffineCalLogLoss
from ..core import Task
import numpy as np
from os import path
import pickle




class DiscriminativeModelCalibration(Task):

    def __init__(self,model):

        modelOutputs = self.load("00_train_system/results/"+model)
        self.model = model
        self.priors = None #podemos probar cambiando esto
        self.targetsTraining = modelOutputs["train"]["labels"]
        self.logPosteriorsTraining = modelOutputs["train"]["logprobs"]
        self.logPosteriorsValidation = modelOutputs["valid"]["logprobs"]

    def run(self):
        calibratedLogPosteriors, parameters = calibrate(self.logPosteriorsTraining, self.targetsTraining, self.logPosteriorsValidation, AffineCalLogLoss, bias=True, priors=self.priors, quiet=True)
        #cambiando el parámetro bias habilita o deshabilita el temp_scaling
        self.save(calibratedLogPosteriors,"01_calibrate_system/results/"+self.model)
        return {"logprobs":calibratedLogPosteriors}

    def save(self,output,output_dir):
        np.savez_compressed(output_dir+"/results.npz",logprobs=output)

    def load(self,output_dir):
        with open(output_dir+"/results.pkl", 'rb') as f:
            data = pickle.load(f)
        return data



