"""
https://www.tensorflow.org/tutorials/quickstart/beginner
Disable GPUs - not all nodes used for testing have GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = ''
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)
import os
import yaml
import logging
import logging.config
import argparse
from enum import Enum
from typing import List
import numpy as np
import torch


logger = logging.getLogger(__name__)


class Task(str, Enum):
    DownloadData = 'download'
    Train = 'train'


def download(task_args: List[str]) -> None:
    """ Task: download.
    Input parameters:
        --data_dir
    """
    logger.info(f"Starting '{Task.DownloadData}' task")
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', '--data-dir', type=str, default=None, help="Path to a dataset file.")
    args = parser.parse_args(args=task_args)

    data_dir = args.data_dir
    if data_dir is None:
        raise ValueError("Data directory is not specified (did you use --data-dir=PATH?)")
    if not data_dir.startswith("/"):
        logger.warning("Data directory seems to be a relative path.")

    logger.info(f"Workspace Data {os.listdir('/workspace/')}")
    # extract the locally copied data and unzip it to data directory
    data_file_name = 'camels'
    data_path = os.path.join(data_dir, data_file_name)
    dir_items = os.listdir(data_path)
    if os.path.isdir(data_path) and len(dir_items) > 0:
        logger.info("Data Already Extracted")
    else:
        os.system(f"unzip /workspace/data/{data_file_name}.zip -d {data_dir}")

    logger.info(f"Data After Extract : {os.listdir(data_path)}")
    logger.info("The '%s' task has been completed.", Task.DownloadData)


def train(task_args: List[str]) -> None:
    """ Task: train.
    Input parameters:
        --data_dir, --log_dir, --model_dir, --parameters_file
    """
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', '--data-dir', type=str, default=None, help="Dataset path.")
    parser.add_argument('--model_dir', '--model-dir', type=str, default=None, help="Model output directory.")
    parser.add_argument('--parameters_file', '--parameters-file', type=str, default=None,
                        help="Parameters default values.")
    args = parser.parse_args(args=task_args)

    with open(args.parameters_file, 'r') as stream:
        parameters = yaml.load(stream, Loader=yaml.FullLoader)

    data_dir = args.data_dir
    model_dir = args.model_dir
    parameters_file = args.parameters_file
    data_file_name = "camels"
    CAMELS_ROOT = os.path.join(data_dir, data_file_name)
    print("HydroDL Data Dir: {}".format(data_dir))
    print("HydroDL Model Dir: {}".format(model_dir))
    print("HydroDL Parameter File: {}".format(parameters_file))
    print("HydroDL Parameters : {}".format(parameters))

    logger.info("Parameters have been read (%s).", args.parameters_file)
    # check the data in the data directory for training programme
    logger.info(f"Data Path : {os.listdir(CAMELS_ROOT)}")

    os.environ['CAMELS_ROOT'] = CAMELS_ROOT
    DAYMET = os.path.join(CAMELS_ROOT, 'basin_mean_forcing/daymet')
    MAURER_EXTENDED = os.path.join(CAMELS_ROOT, 'basin_mean_forcing/maurer_extended')
    STREAMFLOW = os.path.join(CAMELS_ROOT, 'usgs_streamflow')
    ATTRIBUTES = os.path.join(CAMELS_ROOT, 'camels_attributes_v2.0')
    SHAPEFILENAME = 'HCDN_nhru_final_671'
    SHAPEFILE = os.path.join(CAMELS_ROOT, 'camels_timeseries_full_resolution_basin_shapefile')

    # check paths
    assert os.path.exists(DAYMET) == True
    assert os.path.exists(MAURER_EXTENDED) == True
    assert os.path.exists(STREAMFLOW) == True
    assert os.path.exists(ATTRIBUTES) == True

    os.system('git clone https://github.com/cloudmesh/ealstm_regional_modeling.git')
    cur_dir = os.listdir('.')
    logger.info(f'Current Directory : {cur_dir}')

    myregion = 'US'

    import sys
    # Path to the main directory of this repository
    BASE_CODE_DIR = "ealstm_regional_modeling"
    sys.path.append(BASE_CODE_DIR)
    from main import basin_list
    from papercode.plotutils import plot_basins

    shapefile = f"{SHAPEFILE}/{SHAPEFILENAME}"

    # set this to have persistent storage for the results.
    # Otherwise commenting these out would default to the runs directory
    # under the ealstm_regional_modeling
    EXP_DIR = "EALSTM"
    os.makedirs(EXP_DIR, exist_ok=True)

    # Change these parameters for experiments
    batch_size = 256
    dropout = 0.3
    epochs = 3
    hidden_size = 256
    learning_rate = 1e-3
    # loss function could be 'NSE' or 'MSE'
    loss = 'NSE'

    #
    # True to train EALSTM model, False to use normal LSTM model
    use_ealstm = False
    # When not using ealstm, choose whether to use the static features
    # False to NOT use the static feature, True to concat the static features
    # at each time step.
    concat_static = True
    # False to use all characteristics of basins, True to use only
    # the chosen most important ones
    use_partial_attributes = False

    # set run_name
    from datetime import datetime
    import numpy as np
    now = datetime.now()
    day = f"{now.day}".zfill(2)
    month = f"{now.month}".zfill(2)
    hour = f"{now.hour}".zfill(2)
    minute = f"{now.minute}".zfill(2)
    #
    # set seed or use a randomaly generated one
    seed = int(np.random.uniform(low=0, high=1e6))
    #
    # set run_name or use the generated one
    run_name = f'run_{day}{month}_{hour}{minute}_seed{seed}'

    #
    # Setting the environment variables to pass to the training script
    import os
    if EXP_DIR:
        # EXP_DIR=EXP_DIR.replace(" ", "\ ")
        os.environ["LSTM_EXP_DIR"] = f"\"{EXP_DIR}\""
    else:
        os.environ["LSTM_EXP_DIR"] = ""
    os.environ["LSTM_RUN_NAME"] = run_name
    os.environ["LSTM_SEED"] = str(seed)
    os.environ["LSTM_MYREGION"] = myregion
    os.environ["LSTM_BATCH_SIZE"] = str(batch_size)
    os.environ["LSTM_EPOCHS"] = str(epochs)
    os.environ["LSTM_DROPOUT"] = str(dropout)
    os.environ["LSTM_HIDDEN_SIZE"] = str(hidden_size)
    os.environ["LSTM_LEARNING_RATE"] = str(learning_rate)
    if loss == 'MSE':
        os.environ["LSTM_USE_MSE"] = "--use_mse True"
    else:
        os.environ["LSTM_USE_MSE"] = ""
    if not use_ealstm:
        if concat_static:
            os.environ["LSTM_OR_EALSTM"] = "--concat_static True"
        else:
            os.environ["LSTM_OR_EALSTM"] = "--no_static True"
    else:
        os.environ["LSTM_OR_EALSTM"] = ""
    if use_partial_attributes:
        os.environ["LSTM_USE_PARTIAL_ATTRIBUTES"] = "--use_partial_attribs True"
    else:
        os.environ["LSTM_USE_PARTIAL_ATTRIBUTES"] = ""

    train_cmd = f'python3 ealstm_regional_modeling/main.py train --camels_root {CAMELS_ROOT} --exp_dir="$LSTM_EXP_DIR" --seed $LSTM_SEED --region $LSTM_MYREGION --run_name $LSTM_RUN_NAME --epochs $LSTM_EPOCHS --dropout $LSTM_DROPOUT --batch_size $LSTM_BATCH_SIZE --hidden_size $LSTM_HIDDEN_SIZE --learning_rate $LSTM_LEARNING_RATE $LSTM_USE_MSE $LSTM_OR_EALSTM $LSTM_USE_PARTIAL_ATTRIBUTES'
    os.system(train_cmd)


def main():
    """
    hydrodl.py task task_specific_parameters...
    """
    # noinspection PyBroadException
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('mlbox_task', type=str, help="Task for this MLBOX.")
        parser.add_argument('--log_dir', '--log-dir', type=str, required=True, help="Logging directory.")
        ml_box_args, task_args = parser.parse_known_args()

        logger_config = {
            "version": 1,
            "disable_existing_loggers": True,
            "formatters": {
                "standard": {"format": "%(asctime)s - %(name)s - %(threadName)s - %(levelname)s - %(message)s"},
            },
            "handlers": {
                "file_handler": {
                    "class": "logging.FileHandler",
                    "level": "INFO",
                    "formatter": "standard",
                    "filename": os.path.join(ml_box_args.log_dir, f"mlbox_hydrodl_{ml_box_args.mlbox_task}.log")
                }
            },
            "loggers": {
                "": {"level": "INFO", "handlers": ["file_handler"]},
                "__main__": {"level": "NOTSET", "propagate": "yes"},
                "torch": {"level": "NOTSET", "propagate": "yes"}
            }
        }
        logging.config.dictConfig(logger_config)

        if ml_box_args.mlbox_task == Task.DownloadData:
            download(task_args)
        elif ml_box_args.mlbox_task == Task.Train:
            train(task_args)
        else:
            raise ValueError(f"Unknown task: {task_args}")
    except Exception as err:
        logger.exception(err)


if __name__ == '__main__':
    main()
