import os
import sys
from datetime import datetime
import argparse
import numpy as np
from tqdm import tqdm
import pickle

import torch

# local

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

add_path(os.path.abspath('..'))

import pycls.core.builders as model_builder
from pycls.core.config import cfg, dump_cfg
from pycls.datasets.data import Data
import pycls.utils.checkpoint as cu

def argparser():
    parser = argparse.ArgumentParser(description='Passive Learning - Image Classification')
    parser.add_argument('--cfg', dest='cfg_file', help='Config file', required=True, type=str)
    parser.add_argument('--exp-name', dest='exp_name', help='Experiment Name', required=True, type=str)

    return parser


def main(cfg):
    # Using specific GPU
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.GPU_ID)
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    print("Using GPU : {}.\n".format(cfg.GPU_ID))
    
    # Setting up GPU args
    use_cuda = (cfg.NUM_GPUS > 0) and torch.cuda.is_available()
    
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': cfg.DATA_LOADER.NUM_WORKERS, 'pin_memory': cfg.DATA_LOADER.PIN_MEMORY} if use_cuda else {}

    # Getting the output directory ready (default is "/output")
    cfg.OUT_DIR = os.path.join(os.path.abspath('../..'), cfg.OUT_DIR)
    if not os.path.exists(cfg.OUT_DIR):
        os.mkdir(cfg.OUT_DIR)
    # Create "DATASET/MODEL TYPE" specific directory
    dataset_out_dir = os.path.join(cfg.OUT_DIR, cfg.DATASET.NAME, cfg.MODEL.TYPE)
    if not os.path.exists(dataset_out_dir):
        os.makedirs(dataset_out_dir)
    # Creating the experiment directory inside the dataset specific directory 
    # all logs, labeled, unlabeled, validation sets are stroed here 
    # E.g., output/CIFAR10/resnet18/{timestamp or cfg.EXP_NAME based on arguments passed}
    if cfg.EXP_NAME == 'auto':
        now = datetime.now()
        exp_dir = f'{now.year}_{now.month}_{now.day}_{now.hour}{now.minute}{now.second}_feature'
    else:
        exp_dir = cfg.EXP_NAME

    exp_dir = os.path.join(dataset_out_dir, exp_dir)
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)
        print("Experiment Directory is {}.\n".format(exp_dir))
    else:
        print("Experiment Directory Already Exists: {}. Reusing it may lead to loss of old logs in the directory.\n".format(exp_dir))
    cfg.EXP_DIR = exp_dir

    # Save the config file in EXP_DIR
    dump_cfg(cfg)

    # Dataset preparing steps
    print("\n======== PREPARING FEATURE EXTRACTION (TRAIN) DATA ========\n")
    cfg.DATASET.ROOT_DIR = cfg.DATASET.ROOT_DIR # os.path.join(os.path.abspath('../..'), cfg.DATASET.ROOT_DIR)
    data_obj = Data(cfg)
    # test_data, test_size = data_obj.getDataset(save_dir=cfg.DATASET.ROOT_DIR, isTrain=True, isDownload=False) 
    test_data, test_size = data_obj.getDataset(save_dir=cfg.DATASET.ROOT_DIR, isTrain=True) # custom cifar10
    
    print("\nDataset {} Loaded Sucessfully. Total Test Size: {}\n".format(cfg.DATASET.NAME, test_size))

    # Preparing dataloaders for testing
    test_loader = data_obj.getTestLoader(data=test_data, test_batch_size=cfg.TEST.BATCH_SIZE, seed_id=cfg.RNG_SEED)
    
    print("======== FEATURE EXTRACTING ========\n")

    feature_model(test_loader, exp_dir, os.path.join(os.path.abspath('../..'), cfg.TEST.MODEL_PATH), cfg)

    print("...Complete!\n\n")


def feature_model(test_loader, exp_dir, checkpoint_file, cfg, cur_episode=0):

    model = model_builder.build_model(cfg)
    model = cu.load_checkpoint(checkpoint_file, model)
    
    fe_epoch(test_loader, exp_dir, model, cur_episode)


@torch.no_grad()
def fe_epoch(test_loader, exp_dir, model, cur_epoch):
    """Evaluates the model on the test set."""
    if torch.cuda.is_available():
        model.cuda()
    # Enable eval mode
    model.eval()

    feature_list = []
    class_list = []
    image_index_list = []

    for cur_iter, (inputs, labels, image_index) in enumerate(tqdm(test_loader, desc="Test Data")):
        with torch.no_grad():
            inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
            inputs = inputs.type(torch.cuda.FloatTensor)
            feature, _ = model(inputs)
            feature = torch.squeeze(feature)
            
            feature = feature.tolist()
            feature_list.append(feature)

            labels = labels.tolist()
            class_list.append(labels[0])

            image_index = image_index.tolist()
            image_index_list.append(image_index[0])

    with open("{}/feature.pkl".format(exp_dir), 'wb') as f :
        pickle.dump(feature_list, f)
    with open("{}/labels.pkl".format(exp_dir), 'wb') as f :
        pickle.dump(class_list, f)
    with open("{}/image_index.pkl".format(exp_dir), 'wb') as f :
        pickle.dump(image_index_list, f)


if __name__ == "__main__":
    cfg.merge_from_file(argparser().parse_args().cfg_file)
    cfg.EXP_NAME = argparser().parse_args().exp_name
    main(cfg)
