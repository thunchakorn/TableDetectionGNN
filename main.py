from coco_parser import GraphCOCODataset
from torch_geometric.data import DataLoader
import time
from sklearn.metrics import classification_report, confusion_matrix
from model import ResGraph
from loss import WeightedFocalLoss
import torch
from train_utils import *
from pytictoc import TicToc


train_set = GraphCOCODataset('findoc-dataset', ann_file_rpath='results/train.json')
test_set = GraphCOCODataset('findoc-dataset', ann_file_rpath='results/test.json')

train_num = int(len(train_set)*0.9)
val_set = train_set[train_num:]
train_set = train_set[:train_num]

train_loader = DataLoader(train_set, batch_size = 2, shuffle=True, )
val_loader = DataLoader(val_set, batch_size = 2, shuffle=False, )
test_loader = DataLoader(test_set, batch_size = 2, shuffle=False, )

model = ResGraph(772, 256, 128, train_set.num_classes).to(device=try_gpu())
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
criterion = WeightedFocalLoss(gamma=2)
edge_criterion = WeightedFocalLoss(gamma=2)



best_val_loss = None
best_state = None
best_epoch = None
t = TicToc()

for epoch in range(1, 50):
    print(f'_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_ \n epoch: {epoch}')
    t.tic()
    total_train_loss = train(train_loader)
    t.toc('training took')
    with torch.no_grad():
        total_val_loss = loss_monitor(val_loader)
        # train_acc = test(train_loader)

    if best_val_loss is None or total_val_loss < best_val_loss:
        best_val_loss = total_val_loss
        best_epoch = epoch
        best_state = model.state_dict()
        print('+++++++++++++++++++++++++++++++++++++++TABLE+++++++++++++++++++++++++++++++++')
        print('train')
        train_table_acc = test_table(train_loader)
        print('test')
        test_table_acc = test_table(test_loader)

    print(f'''Epoch: {epoch:03d} \n,
    Train total Loss: {total_train_loss:.4f},\n
    Val Total Loss: {total_val_loss},
    Train Tabel ACC {train_table_acc:4f},\n
    Test table ACC {test_table_acc},
    Time: {time_use}''')


print('best epoch is ', best_epoch, 'best val loss = ', best_val_loss)
model.load_state_dict(best_state)
save_path = 'best_weight.pth'
torch.save(model.state_dict(), save_path)
