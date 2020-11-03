import torch
import numpy as np
import cv2
from sklearn.metrics import classification_report, confusion_matrix
import os
import os.path as osp

def try_gpu(i=0):  #@save
    """Return gpu(i) if exists, otherwise return cpu()."""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def train(loader, model, optimizer, criterion, edge_criterion):
    loss_rec = []
    edge_loss_rec = []
    for i, data in enumerate(loader):
        data = data.to(device=try_gpu())
        model.train()
        optimizer.zero_grad()  # Clear gradients.
        out, edge_predict = model(data)  # Perform a single forward pass.
        loss = criterion(out, data.y)  # Compute loss from nodes classification.
        edge_loss = edge_criterion(edge_predict, data.edge_label) # Compute loss from edge classification.
        loss_rec.append(loss.item())
        edge_loss_rec.append(edge_loss.item())
        loss += edge_loss
        loss.backward()  # Derive gradients.

        optimizer.step()  # Update parameters based on gradients.

        # print(f'batch {i}th')
    return np.mean(loss_rec) + np.mean(edge_loss_rec)

def loss_monitor(loader, model, optimizer, criterion, edge_criterion):
    loss_rec = []
    edge_loss_rec = []
    for i, data in enumerate(loader):
        data = data.to(device=try_gpu())
        model.eval()
        optimizer.zero_grad()  # Clear gradients.
        out, edge_predict = model(data)  # Perform a single forward pass.
        loss = criterion(out, data.y)  # Compute loss from nodes classification.
        edge_loss = edge_criterion(edge_predict, data.edge_label) # Compute loss from edge classification.
        loss_rec.append(loss.item())
        edge_loss_rec.append(edge_loss.item())
        # print(f'batch {i}th')
    return np.mean(loss_rec) + np.mean(edge_loss_rec)

def test_table(loader, model):
    model.eval()
    correct = []
    gt_rec = torch.tensor([], device='cpu')
    pred_rec = torch.tensor([])
    for data in loader:  # Iterate in batches over the training/test dataset.
        data = data.to(device=try_gpu())
        out, _ = model(data)
        data = data.to(device='cpu')
        out = out.to(device = 'cpu')
        pred = out.argmax(dim=1)  # Use the class with highest probability.

        pred[(pred != 8) & (pred != 1)] = 0
        pred[(pred == 8)] = 1
        
        data.y[(data.y != 8) & (data.y != 1)] = 0
        data.y[(data.y == 8)] = 1

        gt_rec = torch.cat((gt_rec,data.y))
        pred_rec = torch.cat((pred_rec, pred))

        test_correct = pred == data.y  # Check against ground-truth labels.
        test_acc = int(test_correct.sum()) / int(len(data.y))  # Derive ratio of correct predictions.
        correct.append(test_acc)
    print(classification_report(gt_rec.type(torch.int64), pred_rec))
    print(confusion_matrix(gt_rec.type(torch.int64), pred_rec))
    return np.mean(correct), classification_report(gt_rec.type(torch.int64), pred_rec, output_dict = True)

def test(loader):
    model.eval()
    correct = []
    gt_rec = torch.tensor([])
    pred_rec = torch.tensor([])
    for data in loader:  # Iterate in batches over the training/test dataset.
        out, _ = model(data)  
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        gt_rec = torch.cat((gt_rec,data.y))
        pred_rec = torch.cat((pred_rec, pred))
        test_correct = pred == data.y  # Check against ground-truth labels.
        test_acc = int(test_correct.sum()) / int(len(data.y))  # Derive ratio of correct predictions.
        correct.append(test_acc)
    print(classification_report(gt_rec.type(torch.int64), pred_rec))
    print(confusion_matrix(gt_rec.type(torch.int64), pred_rec))
    return np.mean(correct)

def test_(data):
    model.eval()
    out, _ = model(data)
    pred = out.argmax(dim=1)
    test_correct = pred == data.y  # Check against ground-truth labels.
    test_acc = int(test_correct.sum()) / int(len(data.y))  # Derive ratio of correct predictions.
    return test_acc

def visualising(model, save_dir, loader):
    if not osp.isdir(save_dir):
        os.mkdir(save_dir)
    
    model.eval()
    for data in loader:
        img = cv2.imread(data.image_path[0])
        h, w, c = img.shape
        data = data.to(device=try_gpu())
        out, pred_edge = model(data)
        data = data.to(device='cpu')
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        pred_edge = pred_edge.argmax(dim=1)

        pred[(pred != 8) & (pred != 1)] = 0
        pred[(pred == 8)] = 1

        for g, p in zip(data.x, pred):
            xmin_norm, ymin_norm, xmax_norm, ymax_norm = g[:4]
            xmin = xmin_norm * w
            xmax = xmax_norm * w
            ymin = ymin_norm * h
            ymax = ymax_norm * h

            if p == 1:
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0,0,255), 2, 1)
            else:
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255,0,0), 2, 1)

        for pe, src, dst in zip(pred_edge, data.edge_index[0], data.edge_index[1]):

            xmin_norm, ymin_norm, xmax_norm, ymax_norm = data.x[src][:4]
            xmin_src = xmin_norm * w
            xmax_src = xmax_norm * w
            ymin_src = ymin_norm * h
            ymax_src = ymax_norm * h
            xcen_src = np.mean([xmin_src, xmax_src])
            ycen_src = np.mean([ymin_src, ymax_src])

            xmin_norm, ymin_norm, xmax_norm, ymax_norm = data.x[dst][:4]
            xmin_dst = xmin_norm * w
            xmax_dst = xmax_norm * w
            ymin_dst = ymin_norm * h
            ymax_dst = ymax_norm * h
            xcen_dst = np.mean([xmin_dst, xmax_dst])
            ycen_dst = np.mean([ymin_dst, ymax_dst])

            cv2.circle(img, (xcen_src, ycen_src), 2, (255,0,0), 2)
            cv2.circle(img, (xcen_dst, ycen_dst), 2, (255,0,0), 2)
            if pe == 1:
                cv2.line(img, 
                        (xcen_src, ycen_src), 
                        (xcen_dst, ycen_dst), 
                        (0,255,0), 1)
            else:
                cv2.line(img, 
                        (xcen_src, ycen_src), 
                        (xcen_dst, ycen_dst), 
                        (255,0,255), 1)                
                    
        save_path = osp.join(save_dir, osp.split(data.image_path[0])[-1])
        cv2.imwrite(save_path, img)
        