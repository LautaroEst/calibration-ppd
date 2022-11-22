

n_jobs = 1


pipeline = [
    ("Calibration", {
        "task": "DiscriminativeModelCalibration",
        "input": {"model":"bert"},
        "output": "calibratedLogPosteriors"
    })
]

runs = {0: pipeline}