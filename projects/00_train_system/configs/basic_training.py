
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

    if model == "bert":
        model_init_config = {
            "task": "LoadPretrainedModel",
            "input": None,
            "output": "model",
            "model": "AutoModelForSequenceClassification",
            "source": "bert-base-uncased",
            "num_labels": 1
        }
    elif model == "cbow":
        model_init_config = {
            "task": "InitModelFromTokenizer",
            "input": {"tokenizer": "tokenizer"},
            "output": "model",
            "name": "CBOW",
            "hidden_size": 400,
            "output_size": 1,
            "random_seed": random_seed
        }

    return [
        ("Load Quora Dataset", {
            "task": "LoadQuoraDataset",
            "input": None,
            "output": "qqp_data",
            "cache": True
        }),
        ("Load Twitter Dataset", {
            "task": "LoadTwitterDataset",
            "input": None,
            "output": "twitter_data",
            "cache": True
        }),
        ("Load tokenizer", {
            "task": "LoadPretrainedTokenizer",
            "input": None,
            "output": "tokenizer",
            "model": "AutoTokenizer",
            "source": "bert-base-uncased",
            "cache": True
        }),
        ("Encode dataset", {
            "task": "CreateDynamicPaddingDataloader",
            "input": {
                "tokenizer": "tokenizer", 
                "model_train": "qqp_data.model_train", 
                "calibration_train": "qqp_data.calibration_train", 
                "validation": "qqp_data.validation", 
                "qqp_test": "qqp_data.test",
                "twitter_test": "twitter_data.test"
            },
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
        ("Load metric", {
            "task": "LoadMetrics",
            "input": None,
            "output": "metrics",
            "accuracy": {},
            "fscore": {"beta": 1, "average": "macro"},
            "cache": True
        }),
        ("Load loss function", {
            "task": "LoadLossFunction",
            "input": None,
            "output": "loss_function",
            "name": "BCEWithLogitsLoss",
            "cache": True
        }),
        ("Init Model", model_init_config),
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
                "train_data": "encoded_data.model_train",
                "validation_data": "encoded_data.validation",
                "optimizer": "optimizer",
                "metrics": "metrics"
            },
            "output": "pl_model",
            "enable_checkpointing": True,
            "min_epochs": 3,
            "max_epochs": max_epochs,
            "accelerator": "gpu",
            "devices": 1,
            "val_check_interval": 400,
            "log_every_n_steps": 50
        }),
        ("Evaluation of the model", {
            "task": "MakePredictions",
            "input": {
                "model": "model",
                "model_train": "encoded_data.model_train", 
                "calibration_train": "encoded_data.calibration_train", 
                "validation": "encoded_data.validation", 
                "qqp_test": "encoded_data.qqp_test",
                "twitter_test": "encoded_data.twitter_test"
            },
            "device": "cuda:0",
            "output": "results"
        })
    ]


random_seed = 72435821
runs = {
    "cbow": make_pipeline(
        model = "cbow",
        max_epochs = 3,
        batch_size = 32,
        learning_rate = 1e-3,
        gradient_clip = None,
        weight_decay = 0.0,
        random_seed = random_seed
    ),
    "bert": make_pipeline(
        model = "bert",
        max_epochs = 1,
        batch_size = 16,
        learning_rate = 2e-5,
        gradient_clip = 1.0,
        weight_decay = 0.0,
        random_seed = random_seed
    )
}
