from coco_parser import GraphCOCODataset
from torch_geometric.data import DataLoader
from model import ResGraph
from loss import WeightedFocalLoss
import torch
from train_utils import *
from pytictoc import TicToc
import os

from config_parser import ConfigTrain
import mlflow 
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GNNTableDetection")

def main(ConfigTrain):
    train_set_config = ConfigTrain.Data['train_set']
    test_set_config = ConfigTrain.Data['test_set']
    train_set = GraphCOCODataset(root=train_set_config['root'], dataset_name = train_set_config['dataset_name'])
    test_set = GraphCOCODataset(root=test_set_config['root'], dataset_name = test_set_config['dataset_name'])
    
    train_set_ratio = 1 - train_set_config['val_size']
    train_num = int(len(train_set)*train_set_ratio)
    val_set = train_set[train_num:]
    train_set = train_set[:train_num]
    train_loader = DataLoader(train_set, batch_size = train_set_config['batch_size'], shuffle=True, )
    val_loader = DataLoader(val_set, batch_size = train_set_config['batch_size'], shuffle=False, )
    test_loader = DataLoader(test_set, batch_size = test_set_config['batch_size'], shuffle=False, )

    model_config = ConfigTrain.Model
    optim_config = ConfigTrain.Optim
    train_config = ConfigTrain.Train

    model = ResGraph(num_class = train_set.num_classes, **model_config).to(device=try_gpu())
    optimizer = torch.optim.Adam(model.parameters(), **optim_config)
    criterion = WeightedFocalLoss(gamma=2)
    edge_criterion = WeightedFocalLoss(gamma=2)

    best_val_loss = None
    best_state = None
    best_epoch = None
    t = TicToc()
    logger.info('Training')
    for epoch in range(train_config['epoch']):
        logger.info(f'++++++++++++ epoch: {epoch} of {train_config["epoch"]} ++++++++++++++++++')
        t.tic()
        total_train_loss = train(train_loader, model, optimizer, criterion, edge_criterion)
        t.toc('training took')
        with torch.no_grad():
            total_val_loss = loss_monitor(val_loader, model, optimizer, criterion, edge_criterion)
            # train_acc = test(train_loader)

        if best_val_loss is None or total_val_loss < best_val_loss:
            best_val_loss = total_val_loss
            best_epoch = epoch
            best_state = model.state_dict()
            logger.info('$$$$$$$$$$$$$$$$ TABLE $$$$$$$$$$$$$$$$$$')
            logger.info('table result for train set')
            train_table_acc, _ = test_table(train_loader, model)
            logger.info('table result for test set')
            test_table_acc, _ = test_table(test_loader, model)

        logger.info(f'''Train total Loss: {total_train_loss:.4f},\n
        Val Total Loss: {total_val_loss},
        Train Tabel ACC {train_table_acc:4f},\n
        Test table ACC {test_table_acc}''')


    logger.info(f'best epoch is {best_epoch} best val loss = {best_val_loss}')
    model.load_state_dict(best_state)
    save_path = '/output/best_weight.pth'
    torch.save(model.state_dict(), save_path)

    _, metrics = test_table(test_loader, model)
    mlflow.log_metric('acc', metrics['accuracy'])
    mlflow.log_metrics({'0_'+k:v for k,v in metrics['0'].items()})
    mlflow.log_metrics({'1_'+k:v for k,v in metrics['1'].items()})

    mlflow.log_artifact(save_path)
    test_loader = DataLoader(test_set, batch_size = 1, shuffle=False)
    logger.info('visualising')
    visualising(model, save_dir = '/output/plot', loader = test_loader)
    mlflow.log_artifacts('/output/plot')


if __name__ == "__main__":
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
    mlflow.set_experiment(os.getenv('MLFLOW_EXPERIMENT_NAME'))
    output_dir = '/output/'
    if not osp.isdir(output_dir):
        os.mkdir(output_dir)
    with mlflow.start_run():
        mlflow.log_artifact('config/config.yaml')
        main(ConfigTrain)