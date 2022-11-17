from copy import deepcopy

from ..core import Task
import datasets
from datasets import DatasetDict, Dataset, load_dataset
from typing import Union



class LoadQuoraDataset(Task):

    def __init__(self):
        self.random_seed = 17283
        self.opt = "split_validation" # "split_validation" (opción 1) | "split_train" (opción 2)

    def run(self):
        qqp_data = load_dataset("glue","qqp")

        if self.opt == "split_validation":

            qqp_train_splitted = qqp_data["train"].train_test_split(test_size=0.2,seed=self.random_seed,stratify_by_column="label")
            qqp_val_splitted = qqp_data["validation"].train_test_split(test_size=0.2,seed=self.random_seed,stratify_by_column="label")

            data = {
                "training": DatasetDict({
                    "train": qqp_train_splitted["train"],
                    "val": qqp_train_splitted["test"],
                    "test": qqp_data["test"]
                }),
                "calibration": DatasetDict({
                    "train": qqp_val_splitted["train"],
                    "val": qqp_val_splitted["test"],
                    "test": deepcopy(qqp_data["test"])
                })
            }
        elif self.opt == "split_train":
            qqp_train_splitted = qqp_data["train"].train_test_split(test_size=0.2,seed=self.random_seed,stratify_by_column="label")

            data = {
                "training": DatasetDict({
                    "train": qqp_train_splitted["train"],
                    "validation": qqp_data["validation"],
                    "test": qqp_data["test"]
                }),
                "calibration": DatasetDict({
                    "train": qqp_train_splitted["test"],
                    "validation": deepcopy(qqp_data["validation"]),
                    "test": deepcopy(qqp_data["test"])
                })
            }
        else:
            raise ValueError("Opción no válida")

        return data

    def save(self,output: Union[Dataset,DatasetDict],output_dir):
        output.save_to_disk(output_dir)


    def load(self,output_dir):
        data = datasets.load_from_disk(output_dir)
        return data



class LoadTwitterDataset(Task):

    def __init__(self):
        self.random_seed = 17283

    def run(self):
        ## TODO: descargar y dividir este dataset
        data = {
            "training": {
                "train": [],
                "val": [],
                "test": []
            },
            "calibration": {
                "train": [],
                "val": [],
                "test": []
            }
        }
        return data

    def save(self,output: Union[datasets.Dataset,datasets.DatasetDict],output_dir):
        output.save_to_disk(output_dir)


    def load(self,output_dir):
        data = datasets.load_from_disk(output_dir)
        return data


if __name__ == "__main__":
    task = LoadQuoraDataset()
    data = task.run()
    import pdb; pdb.set_trace()