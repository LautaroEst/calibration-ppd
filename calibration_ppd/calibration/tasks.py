from calibration import calibrate,AffineCalLogLoss
from ..core import Task



class DiscriminativeModelCalibration(Task):

    def __init__(self,dataset):
        self.priors = None #podemos probar cambiando esto
        self.targetsTraining = None
        self.logPosteriorsTraining = None
        self.logPosteriorsValidation = None

    def run(self):
        calibratedLogPosteriors, parameters = calibrate(self.logPosteriorsTraining, self.targetsTraining, self.logPosteriorsValidation, AffineCalLogLoss, bias=True, priors=self.priors, quiet=True)
        #cambiando el parámetro bias habilita o deshabilita el temp_scaling
        return calibratedLogPosteriors

    def save(self,output,output_dir):
        #en qué formato guardo los resultados?
        pass

    def load(self,output_dir):
        pass



