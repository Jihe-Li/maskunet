import os
import argparse
from yacs.config import CfgNode as CN

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='configs/completion.yaml')
parser.add_argument('--process_dataset', help='whether to process data')
args = parser.parse_args()


with open(args.config, 'r') as fid:
    CONFIGS = CN.load_cfg(fid)
