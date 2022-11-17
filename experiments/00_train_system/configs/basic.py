
n_jobs = 1


pipeline = [
    ("Load Dataset", {
        "task": "LoadQuoraDataset",
        "input": None,
        "output": "data",
        "cache": True
    })
]

runs = {0: pipeline}
