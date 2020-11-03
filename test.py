from coco_parser import GraphCOCODataset
from torch_geometric.data import DataLoader
from model import ResGraph
import torch
from train_utils import *
from pytictoc import TicToc
import os
import mlflow

from config_parser import ConfigTest
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GNNTableDetection")

def main(ConfigTest):
    test_set_config = ConfigTest.Data
    test_set = GraphCOCODataset(**test_set_config)
    test_loader = DataLoader(test_set, batch_size = 1, shuffle=False)

    model_config = ConfigTest.Model
    weight_path = ConfigTest.Weight['path']
    model = ResGraph(num_class = test_set.num_classes, **model_config).to(device=try_gpu())
    logger.info('load weight')
    model.load_state_dict(torch.load(weight_path))
    logger.info('inferencing')
    _, metrics = test_table(test_loader, model)
    mlflow.log_metric('acc', metrics['accuracy'])
    mlflow.log_metrics({'0_'+k:v for k,v in metrics['0'].items()})
    mlflow.log_metrics({'1_'+k:v for k,v in metrics['1'].items()})

    test_loader = DataLoader(test_set, batch_size = 1, shuffle=False)
    visualising(model, save_dir = '/output/plot', loader = test_loader)
    mlflow.log_artifacts('/output/plot')


if __name__ == "__main__":
    output_dir = '/output/'
    if not osp.isdir(output_dir):
        os.mkdir(output_dir)
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
    mlflow.set_experiment(os.getenv('MLFLOW_EXPERIMENT_NAME'))
    with mlflow.start_run():
        mlflow.log_artifact('config/config_test.yaml')
        main(ConfigTest)