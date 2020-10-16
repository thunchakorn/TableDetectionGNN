from RVL_parser import RVL_Dataset
from torch_geometric.data import DataLoader
import time
from sklearn.metrics import classification_report, confusion_matrix
from model import ResGraph
from loss import WeightedFocalLoss
import torch
from train_utils import *


rvl_dataset = RVL_Dataset('./RVL_Dataset', dir = './RVL_Dataset/dataset')
train_set = rvl_dataset[:362]
val_set = rvl_dataset[362:362+52]
test_set = rvl_dataset[362+52:]
train_loader = DataLoader(train_set, batch_size = 8, shuffle=True, )
val_loader = DataLoader(val_set, batch_size = 8, shuffle=False, )
test_loader = DataLoader(test_set, batch_size = 8, shuffle=False, )

model = ResGraph(772, 64, 32, 6)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
criterion = WeightedFocalLoss(gamma=2)
edge_criterion = WeightedFocalLoss(gamma=2)



nl = None
for epoch in range(1, 300):
    print(f'++++++++++++++++++++++++++++++++++++++++++ \n epoch: {epoch}')
    start = time.time()
    loss, edge_loss = train(train_loader)

    time_use = time.time() - start
    with torch.no_grad():
        nl, el = loss_monitor(val_loader)
        train_acc = test(train_loader)
    
        train_table_acc = test_table(train_loader)
        val_table_acc = test_table(val_loader)
        test_table_acc = test_table(test_loader)

    print(f'''Epoch: {epoch:03d}, Train Loss: {loss:.4f},
    Train Edge Loss: {edge_loss:.4f},\n
    Val Loss: {nl},
    Val edge Loss: {el},\n
    Train Acc: {train_acc:.4f},
    Train Tabel ACC {train_table_acc:4f},\n
    Test table ACC {test_table_acc},
    Time: {time_use}''')