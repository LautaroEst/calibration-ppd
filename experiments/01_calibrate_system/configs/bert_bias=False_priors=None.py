

n_jobs = 1


def make_pipeline(
    model_name,
    bias,
    priors,
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
            "priors": priors,
            "calibration_loss": calibration_loss
        })
    ]

pipeline = make_pipeline(
    model_name = "bert",
    bias = False, # cambiando el parámetro bias habilita o deshabilita el temp_scaling
    priors = None, # podemos probar cambiando esto
    calibration_loss = "affine_log"
)
runs = {0: pipeline}