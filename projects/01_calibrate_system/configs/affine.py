

from itertools import product


n_jobs = 1


def make_pipeline(
    model_name,
    training_prior,
    evaluation_prior,
    calibration_loss
):
    return [
        ("Loading Predictions", {
            "task": "LoadModelPredictions",
            "input": None,
            "path": f"../00_train_system/results/{model_name}/09_Evaluation of the model/results.pkl",
            "output": "results"
        }),
        ("Calibration", {
            "task": "DiscriminativeModelCalibration",
            "input": {"model_outputs": "results"},
            "output": "calibratedLogPosteriors",
            "training_prior": training_prior,
            "evaluation_prior": evaluation_prior,
            "calibration_loss": calibration_loss,
            "epochs": 10000,
            "lr": 5e-3,
            "device": "cuda:0",
            "run_on_test": True
        })
    ]


models = ["cbow", "bert"]
training_priors = [0.1, 0.36] #[0.1, 0.2, 0.36, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
evaluation_priors = [0.1, 0.36]
calibration_losses = ["affine_log"]

runs = {}
# for model_name,bias,prior_class1,calibration_loss in product(models,biases,priors,calibration_losses):
#     runs[f"{model_name}_{bias}_{prior_class1}_{calibration_loss}"] = make_pipeline(model_name,bias,prior_class1,calibration_loss)

for model_name, tp, ep, calibration_loss in product(models,training_priors ,evaluation_priors, calibration_losses):
    runs[f"{model_name}_{tp}_{ep}_{calibration_loss}"] = make_pipeline(model_name, tp, ep, calibration_loss)