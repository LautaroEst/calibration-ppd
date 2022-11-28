import argparse
import json
import os
import requests
import calibration_ppd as cppd
from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings",type=str,help="Settings JSON file")
    args = parser.parse_args()
    return args

def read_settings(file):
    with open(file,"r") as f:
        settings = json.load(f)
    return settings


def check_and_create_data_dir():
    if not os.path.exists(cppd.DATASETS_DIR):
        print("Creating datasets directory...")
        os.mkdir(cppd.DATASETS_DIR)
    if not os.path.exists(cppd.MODELS_DIR):
        print("Creating models directory...")
        os.mkdir(cppd.MODELS_DIR)


def parse_actions(*settings):
    for setting in settings:
        action = setting.pop("action")
        print(action)
        if action == "create_link_to_dataset":
            true_path = setting.pop("true_path")
            create_link(true_path,cppd.DATASETS_DIR)
        # elif action == "create_link_to_model":
        #     true_path = setting.pop("true_path")
        #     create_link(true_path,dm.MODELS_DIR)
        elif action == "download_huggingface_dataset":
            download_huggingface_dataset(**setting)
        elif action == "download_file":
            url = setting.pop("url")
            # save_in = setting.pop("save_in")
            get_file_from_url(url,cppd.DATASETS_DIR)
        else:
            raise ValueError(f"Action {action} not supported")


def create_link(true_path,target_path):
    new_path = os.path.join(target_path,os.path.basename(os.path.normpath(true_path)))
    try:
        os.symlink(true_path,new_path)
        print(f"Created link for {true_path}...")
    except FileExistsError:
        print(f"{true_path} already has a link")


def download_huggingface_dataset(**kwargs):
    load_dataset(
        cache_dir=cppd.DATASETS_DIR,
        path=kwargs.pop("path"),
        name=kwargs.pop("name",None)
    )


def get_file_from_url(url,target_path):
    r = requests.get(url)
    full_filename = os.path.join(target_path,os.path.basename(url))
    with open(full_filename,"w") as f:
        f.write(r.text)
    

def create_results_dir(file):
    experiment_name = file.split("/")[0]
    results_dir = os.path.join(cppd.PROJECTS_DIR,experiment_name,"results")
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    else:
        print("Results dir already exists")


def main():
    args = parse_args()
    settings = read_settings(args.settings)
    check_and_create_data_dir()
    parse_actions(*settings)
    create_results_dir(args.settings)


if __name__ == "__main__":
    main()