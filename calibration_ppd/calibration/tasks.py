from calibration import calibrate,AffineCalLogLoss
from ..core import Task
import numpy as np
from os import path


class DiscriminativeModelCalibration(Task):

    def __init__(self):
        modelOutputs = self.load("00_train_system/results")

        self.priors = None #podemos probar cambiando esto
        self.targetsTraining = modelOutputs["train"]["labels"]
        self.logPosteriorsTraining = modelOutputs["train"]["logprobs"]
        self.logPosteriorsValidation = modelOutputs["valid"]["logprobs"]

    def run(self):
        calibratedLogPosteriors, parameters = calibrate(self.logPosteriorsTraining, self.targetsTraining, self.logPosteriorsValidation, AffineCalLogLoss, bias=True, priors=self.priors, quiet=True)
        #cambiando el parámetro bias habilita o deshabilita el temp_scaling
        self.save(calibratedLogPosteriors,"01_calibrate_system/results")
        return {"logprobs":calibratedLogPosteriors}

    def save(self,output,output_dir):
        np.savez_compressed(output_dir+"/calibratedOutputs.npz",logprobs=output)

    def load(self,output_dir):
        data = np.load(path.abspath(output_dir+"/modelOutputs.npz"))
        return data



