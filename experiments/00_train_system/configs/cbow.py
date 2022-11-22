
n_jobs = 1

def make_pipeline(
    model,
    max_epochs,
    batch_size,
    learning_rate,
    gradient_clip,
    weight_decay,
    random_seed
):
    return [
        ("Load Dataset", {
            "task": "LoadQuoraDataset",
            "input": None,
            "output": "data",
            "cache": True
        }),
        ("Load tokenizer", {
            "task": "LoadPretrainedTokenizer",
            "input": None,
            "output": "tokenizer",
            "model": "AutoTokenizer",
            "source": model,
            "cache": True
        }),
        ("Encode dataset", {
            "task": "CreateDynamicPaddingDataloader",
            "input": {"tokenizer": "tokenizer", "train": "data.training.train", "validation": "data.training.validation", "test": "data.training.test"},
            "output": "encoded_data",
            "text": "question1",
            "text_pair": "question2",
            "labels": "label",
            "cache": True,
            # Tokenizer args:
            "truncation": True,
            "max_length": 256, # largo máximo de una secuencia (truncation tiene que ser True)
            "padding": "longest", # longest | max_length
            # Map args:
            "batch_dataset_mapping": True,
            "dataset_mapping_batch_size": 512,
            # Dataloader args:
            "batch_size": batch_size,
            "random_seed": random_seed,
            "num_workers": 8
        }),
        ("Init Model", {
            "task": "InitTorchModel",
            "input": {"tokenizer": "tokenizer"},
            "output": "model",
            "name": "CBOW",
            "hidden_size": 400,
            "output_size": 1
        }),
        ("Load metric", {
            "task": "LoadMetrics",
            "input": None,
            "output": "metrics",
            "accuracy": {},
            "fscore": {"beta": 1, "average": "macro"}
        }),
        ("Load loss function", {
            "task": "LoadLossFunction",
            "input": None,
            "output": "loss_function",
            "name": "BCEWithLogitsLoss"
        }),
        ("Init optimization procedure", {
            "task": "InitAdamWOptimization",
            "input": {"model": "model"},
            "output": "optimizer",
            "learning_rate": learning_rate,
            "gradient_clip": gradient_clip,
            "weight_decay": weight_decay
        }),
        ("Train model", {
            "task": "TrainSupervisedTorchModel",
            "input": {
                "model": "model", 
                "loss_fn": "loss_function",
                "train_data": "encoded_data.train",
                "validation_data": "encoded_data.validation",
                "optimizer": "optimizer",
                "metrics": "metrics"
            },
            "output": None,
            "enable_checkpointing": True,
            "min_epochs": 1,
            "max_epochs": max_epochs,
            "accelerator": "cpu",
            "devices": 1,
            "val_check_interval": 400,
            "log_every_n_steps": 50
        })
    ]

train_params = dict(
    model = "bert-base-uncased",
    max_epochs = 3,
    batch_size = 32,
    learning_rate = 1e-3,
    gradient_clip = None,
    weight_decay = 0.0,
    random_seed = 72435821
)
runs = {0: make_pipeline(**train_params)}
