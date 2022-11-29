import os

from ..core import Task
from datasets import DatasetDict, Dataset, load_dataset, ClassLabel, load_from_disk, Features, Value
from ..utils import DATASETS_DIR

import torch
from torch.utils.data import RandomSampler, DataLoader

import pandas as pd



class LoadQuoraDataset(Task):

    def __init__(self):
        self.random_seed = 17283

    def run(self):
        qqp_data = load_dataset("glue","qqp",cache_dir=DATASETS_DIR)

        data_splitted = qqp_data["train"].train_test_split(test_size=0.2,stratify_by_column="label")
        data_splitted_splitted = data_splitted["test"].train_test_split(test_size=0.5,stratify_by_column="label")

        data = DatasetDict({
            "model_train": data_splitted["train"],
            "calibration_train": data_splitted_splitted["train"],
            "validation": data_splitted_splitted["test"],
            "test": qqp_data["validation"]
        })

        return data


    def save(self,output,output_dir):
        for key, dataset in output.items():
            path = os.path.join(output_dir,key)
            os.mkdir(path)
            dataset.save_to_disk(path)


    def load(self,output_dir):
        directories = os.listdir(output_dir)
        data = {directory: load_from_disk(os.path.join(output_dir,directory)) for directory in directories}
        return data



class LoadTwitterDataset(Task):

    def __init__(self):
        self.random_seed = 17283

    def run(self):
        with open(os.path.join(DATASETS_DIR,"a.toks"),"r") as f:
            question1 = [s[:-1] for s in f.readlines()]
        with open(os.path.join(DATASETS_DIR,"b.toks"),"r") as f:
            question2 = [s[:-1] for s in f.readlines()]
        with open(os.path.join(DATASETS_DIR,"sim.txt"),"r") as f:
            labels = [s[:-1] for s in f.readlines()]
        dataset = pd.DataFrame.from_dict({"question1": question1, "question2": question2, "label": labels})
        dataset["question1"] = dataset["question1"].astype(str)
        dataset["question2"] = dataset["question2"].astype(str)
        dataset["label"] = dataset["label"].astype(int)
        data = DatasetDict({
            "test": Dataset.from_pandas(dataset,Features.from_dict({
                "question1": {"dtype": "string", "id": None, "_type": "Value"},
                "question2": {"dtype": "string", "id": None, "_type": "Value"},
                "label": {"num_classes": 2, "names": ["not-sim","sim"], "_type": "ClassLabel"}
            }))
        })
        return data

    def save(self,output,output_dir):
        output.save_to_disk(output_dir)


    def load(self,output_dir):
        data = load_from_disk(output_dir)
        return data




class CreateDynamicPaddingDataloader(Task):

    def __init__(self,**kwargs):
        self.text_fields = dict(
            text=kwargs.pop("text",None),
            text_pair=kwargs.pop("text_pair",None),
            text_target=kwargs.pop("text_target",None),
            text_pair_target=kwargs.pop("text_pair_target",None)
        )
        self.labels_column = kwargs.pop("labels")
        self.enconding_args = dict(
            add_special_tokens=kwargs.pop("add_special_tokens",True),
            padding="do_not_pad",
            truncation=kwargs.pop("truncation",False),
            max_length=kwargs.pop("max_length",None),
            stride=kwargs.pop("stride",0),
            is_split_into_words=kwargs.pop("is_split_into_words",False),
            pad_to_multiple_of=None,
            return_tensors=None,
            return_token_type_ids=True,
            return_attention_mask=True,
            return_overflowing_tokens=kwargs.pop("return_overflowing_tokens",False),
            return_special_tokens_mask=kwargs.pop("return_special_tokens_mask",False),
            return_offsets_mapping=kwargs.pop("return_offsets_mapping",False),
            return_length=kwargs.pop("return_length",False),
            verbose=kwargs.pop("verbose",True)
        )

        self.generator = torch.Generator()
        random_seed = kwargs.pop("random_seed",None)
        if random_seed:
            self.generator.manual_seed(int(random_seed))

        self.mapping_args = dict(
            with_indices=kwargs.pop("with_indices",False),
            with_rank=kwargs.pop("with_rank",False),
            input_columns=kwargs.pop("input_columns",None),
            batched=kwargs.pop("batch_dataset_mapping",False),
            batch_size=kwargs.pop("dataset_mapping_batch_size",1000),
            drop_last_batch=kwargs.pop("drop_last_batch",False),
            remove_columns=None,
            keep_in_memory=kwargs.pop("keep_in_memory",False),
            load_from_cache_file=kwargs.pop("load_from_cache_file",None),
            cache_file_name=kwargs.pop("cache_file_name",None),
            writer_batch_size=kwargs.pop("writer_batch_size",1000),
            features=kwargs.pop("features",None),
            disable_nullable=kwargs.pop("disable_nullable",False),
            fn_kwargs=kwargs.pop("fn_kwargs",None),
            num_proc=kwargs.pop("num_proc",None),
            suffix_template=kwargs.pop("suffix_template","_{rank:05d}_of_{num_proc:05d}"),
            new_fingerprint=kwargs.pop("new_fingerprint",None),
            desc=kwargs.pop("desc",None)
        )

        self.batch_size=kwargs.pop("batch_size",2)
        self.num_workers=kwargs.pop("num_workers",0)
        self.padding=kwargs.pop("padding","longest")
        self.max_length=kwargs.pop("max_length",None)
        self.pad_to_multiple_of=kwargs.pop("pad_to_multiple_of",None)
        self.pin_memory=kwargs.pop("pin_memory",False)
        self.drop_last=kwargs.pop("drop_last",False)
        self.timeout=kwargs.pop("timeout",0)
        self.worker_init_fn=kwargs.pop("worker_init_fn",None)
        self.multiprocessing_context=kwargs.pop("multiprocessing_context",None)
        self.generator=kwargs.pop("generator",None)
        self.prefetch_factor=kwargs.pop("prefetch_factor",2)
        self.persistent_workers=kwargs.pop("persistent_workers",False)
        self.pin_memory_device=kwargs.pop("pin_memory_device","")

    def _create_sampler(self,dataset):
        sampler = RandomSampler(dataset,num_samples=len(dataset),generator=self.generator,replacement=False)
        return sampler

    def apply_encoding(self,sample):
        text_fields = {field: sample[textfield] if textfield else None for field, textfield in self.text_fields.items()}
        all_args = {**text_fields,**self.enconding_args}
        return self.tokenizer(**all_args)
        
    def data_collator(self,features):
        return self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length if self.padding == "max_length" else None,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

    def run(self,tokenizer,**datasets):
        self.tokenizer = tokenizer
        dataloaders = {}
        for name, dataset in datasets.items():
            if not self.labels_column == "label":
                dataset = dataset.rename_column(self.labels_column,"label")
            if not isinstance(dataset.features["label"],ClassLabel):
                dataset = dataset.class_encode_column("label")
            self.mapping_args["remove_columns"] = [column for column in dataset.column_names if column not in ["input_ids","attention_mask","token_type_ids","label"]]
            dataset = dataset.map(self.apply_encoding,**self.mapping_args)
            dataloaders[name] = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                sampler=self._create_sampler(dataset),
                batch_sampler=None,
                num_workers=self.num_workers,
                collate_fn=self.data_collator,
                pin_memory=self.pin_memory,
                drop_last=self.drop_last,
                timeout=self.timeout,
                worker_init_fn=self.worker_init_fn,
                multiprocessing_context=self.multiprocessing_context,
                generator=self.generator,
                prefetch_factor=self.prefetch_factor,
                persistent_workers=self.persistent_workers,
                pin_memory_device=self.pin_memory_device
            )
        return dataloaders




if __name__ == "__main__":
    task = LoadQuoraDataset()
    data = task.run()
    import pdb; pdb.set_trace()