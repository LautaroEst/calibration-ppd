import argparse
import json
import os
import requests

ROOT_DIR = os.path.join(os.getcwd().split("calibration-ppd")[0],"calibration-ppd")
DATA_DIR = os.path.join(ROOT_DIR,"data")
SCRIPTS_DIR = os.path.join(ROOT_DIR,"scripts")
EXPERIMENTS_DIR = os.path.join(ROOT_DIR,"experiments")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings",type=str,help="Settings JSON file")
    args = parser.parse_args()
    return args

def read_settings(file):
    with open(file,"r") as f:
        settings = json.load(f)
    return settings

def parse_actions(*settings):
    for setting in settings:
        action = setting.pop("action")
        print(action)
        if action == "create_link":
            true_path = setting.pop("data_true_path")
            create_link(true_path,DATA_DIR)
        elif action == "download_file":
            url = setting.pop("url")
            get_file_from_url(url,DATA_DIR)
        else:
            raise ValueError(f"Action {action} not supported")


def create_link(true_path,target_path):
    new_path = os.path.join(target_path,os.path.basename(os.path.normpath(true_path)))
    try:
        os.symlink(true_path,new_path)
        print(f"Created link for {true_path}...")
    except FileExistsError:
        print(f"{true_path} already has a link")

def get_file_from_url(url,target_path):
    r = requests.get(url)
    full_filename = os.path.join(target_path,os.path.basename(url))
    with open(full_filename,"w") as f:
        f.write(r.text)
    


def main():
    args = parse_args()
    settings = read_settings(args.settings)
    parse_actions(*settings)


if __name__ == "__main__":
    main()
