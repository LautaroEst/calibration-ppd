

n_jobs = 1


pipeline = [
    ("Calibration", {
        "task": "DiscriminativeModelCalibration",
        "input": None,
        "output": "calibratedLogPosteriors"
    })
]

runs = {0: pipeline}