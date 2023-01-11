

from itertools import product
import os


n_jobs = 1


def make_pipeline(
    model_name,
    training_prior,
    evaluation_prior
):
    model = "run00_cbow" if model_name == "cbow" else "run01_bert"
    return [
        ("Loading Predictions", {
            "task": "LoadModelPredictions",
            "input": None,
            "path": f"00_train_system/results/basic_training/{model}/09_Evaluation of the model/results.pkl",
            "output": "results"
        }),
        ("Calibration", {
            "task": "DiscriminativeModelCalibration",
            "input": {"model_outputs": "results"},
            "output": "calibratedLogPosteriors",
            "training_prior": training_prior,
            "evaluation_prior": evaluation_prior,
            "calibration_loss": "affine_log",
            "epochs": 10000,
            "lr": 5e-4,
            "device": "cuda:0",
            "run_on_test": True
        })
    ]


models = ["cbow", "bert"]
training_priors = [0.1, 0.2, 0.36]
evaluation_priors = [0.1, 0.2, 0.36]

runs = {}
for model_name, tp, ep in product(models,training_priors ,evaluation_priors):
    runs[f"{model_name}_{tp}_{ep}"] = make_pipeline(model_name, tp, ep)