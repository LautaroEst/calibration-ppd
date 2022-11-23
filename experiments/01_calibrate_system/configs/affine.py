

from itertools import product


n_jobs = 1


def make_pipeline(
    model_name,
    bias,
    prior_class1,
    calibration_loss
):
    return [
        ("Loading Predictions", {
            "task": "LoadModelPredictions",
            "input": None,
            "path": f"01_calibrate_system/{model_name}_results.pkl",
            "output": "results"
        }),
        ("Calibration", {
            "task": "DiscriminativeModelCalibration",
            "input": {"model_outputs": "results"},
            "output": "calibratedLogPosteriors",
            "bias": bias, 
            "prior_class1": prior_class1,
            "calibration_loss": calibration_loss,
            "epochs": 10000,
            "lr": 5e-3,
            "device": "cuda:0"
        })
    ]


models = ["cbow", "bert"]
biases = [True]
priors = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
calibration_losses = ["affine_log"]

runs = {}
for model_name,bias,prior_class1,calibration_loss in product(models,biases,priors,calibration_losses):
    runs[f"{model_name}_{bias}_{prior_class1}_{calibration_loss}"] = make_pipeline(model_name,bias,prior_class1,calibration_loss)