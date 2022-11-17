
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
        })
    ]

train_params = dict(
    model = "bert-base-uncased",
    max_epochs = 3,
    batch_size = 16,
    learning_rate = 2e-5,
    gradient_clip = 1.0,
    weight_decay = 0.0,
    random_seed = 72435821
)
runs = {0: make_pipeline(**train_params)}
