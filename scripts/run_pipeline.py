import argparse
import os
import time
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import calibration_ppd as cppd

from calibration_ppd.utils import import_configs_objs


def parse_args():
    """Función para parsear los argumentos de línea de comando"""

    # Inicializo el argparse
    parser = argparse.ArgumentParser()

    # Lista de argumentos por línea de comando
    parser.add_argument("--config", help="Config file with the experiment configurations")

    # Convierto a diccionario
    command_line_args = vars(parser.parse_args())
    return command_line_args
    

def run_pipeline(pipeline,run_name,results_dir):

    cache_dir = os.path.join(results_dir,"cache")
    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)

    all_outputs = {}
    for i, (task_name, task_config) in enumerate(pipeline):
        print(f"Running {task_name} task of run {run_name}...")

        cache_task = task_config.pop("cache",False)
        if cache_task:
            task_results_dir = os.path.join(results_dir,"cache",f"{i:02}_{task_name}")
        else:
            task_results_dir = os.path.join(results_dir,f"run{run_name}",f"{i:02}_{task_name}")
        
        if not os.path.exists(task_results_dir):
            os.makedirs(task_results_dir)

            # Task initialization
            task_cls_name = task_config.pop("task")
            task_input_dict = task_config.pop("input")
            task_output_name = task_config.pop("output")
            task_cls = getattr(cppd,task_cls_name)
            task = task_cls(**task_config)
            task.set_output_dir(task_results_dir)
            
            # Configure the task inputs
            inputs = {}
            if task_input_dict is not None:
                for input_key, input_val in task_input_dict.items():
                    if "." in input_val:
                        keys = input_val.split('.')
                        val = all_outputs[keys.pop(0)]
                        for key in keys:
                            val = val[key]
                        inputs[input_key] = val
                    else:
                        inputs[input_key] = all_outputs[input_val]

            # Run the task
            task_output = task.run(**inputs)

            # Get the task outputs
            if task_output_name is not None:
                all_outputs[task_output_name] = task_output
                task.save_output_to_disk(task_output)

        elif cache_task:
            task_cls_name = task_config.pop("task")
            _ = task_config.pop("input")
            task_output_name = task_config.pop("output")
            task_cls = getattr(cppd,task_cls_name)
            task = task_cls(**task_config)
            task.set_output_dir(task_results_dir)
            if task_output_name is not None:
                all_outputs[task_output_name] = task.load_output_from_disk()
            else:
                raise RuntimeError("Cannot load output from a None-returning task")

        else:
            raise RuntimeError("Run directory should not exist if cache is False")

    return run_name


def main(**kwargs):
    """Función principal"""

    # Carga de los objetos del config en un objeto
    config_path = kwargs.pop("config")
    config = import_configs_objs(config_path)

    # Creo el directorio de resultados:
    experiment_name, _, config_name = config_path.split("/")[-3:]
    results_dir = f"{experiment_name}/results/{config_name.split('.')[0]}"
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    # Run pipelines:
    if config.n_jobs > 1:
        for pipeline in config.runs.values():
            for pipe in pipeline:
                if pipe[1].get("cache",False):
                    raise RuntimeError("Caching is not allowed when n_jobs > 1")
        job = (delayed(run_pipeline)(pipeline,f"{idx:02}_{run_name}",results_dir) for idx, (run_name, pipeline) in enumerate(tqdm(config.runs.items())))
        n_jobs = getattr(config,"n_jobs",1)
        pipelines_generator = Parallel(n_jobs=n_jobs)(job)
    else:
        pipelines_generator = (run_pipeline(pipeline,f"{idx:02}_{run_name}",results_dir) for idx, (run_name, pipeline) in enumerate(tqdm(config.runs.items())))

    tic = time.time()
    print("Start running...")
    for run_name in pipelines_generator:
        now = time.time() - tic
        print(f"Time of run {run_name}: {int(now//3600):02}h{int(now//60):02}m{int(now)%60:02}s")


if __name__ == "__main__":

    # 1) Lectura de los argumentos
    kwargs = parse_args()

    # 2) Función principal
    main(**kwargs)