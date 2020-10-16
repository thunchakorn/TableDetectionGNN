import torch
import numpy as np

def train(loader):
    loss_rec = []
    edge_loss_rec = []
    for i, data in enumerate(loader):
        model.train()
        optimizer.zero_grad()  # Clear gradients.
        out, edge_predict = model(data)  # Perform a single forward pass.
        loss = criterion(out, data.y)  # Compute loss from nodes classification.
        edge_loss = edge_criterion(edge_predict, data.edge_label) # Compute loss from edge classification.

        loss.backward(retain_graph = True)  # Derive gradients.
        edge_loss.backward()

        optimizer.step()  # Update parameters based on gradients.
        loss_rec.append(loss.item())
        edge_loss_rec.append(edge_loss.item())
        # print(f'batch {i}th')
    return np.mean(loss_rec), np.mean(edge_loss_rec)

def loss_monitor(loader):
    loss_rec = []
    edge_loss_rec = []
    for i, data in enumerate(loader):
        model.train()
        optimizer.zero_grad()  # Clear gradients.
        out, edge_predict = model(data)  # Perform a single forward pass.
        loss = criterion(out, data.y)  # Compute loss from nodes classification.
        edge_loss = edge_criterion(edge_predict, data.edge_label) # Compute loss from edge classification.
        loss_rec.append(loss.item())
        edge_loss_rec.append(edge_loss.item())
        # print(f'batch {i}th')
    return np.mean(loss_rec), np.mean(edge_loss_rec)

def test_table(loader):
    #table class position is 3
    model.eval()
    correct = []
    gt_rec = torch.tensor([])
    pred_rec = torch.tensor([])
    for data in loader:  # Iterate in batches over the training/test dataset.
        out, _ = model(data)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        pred[pred != 3] = 0
        pred[pred == 3] = 1
        data.y[data.y != 3] = 0
        data.y[data.y == 3] = 1 
        gt_rec = torch.cat((gt_rec,data.y))
        pred_rec = torch.cat((pred_rec, pred))

        test_correct = pred == data.y  # Check against ground-truth labels.
        test_acc = int(test_correct.sum()) / int(len(data.y))  # Derive ratio of correct predictions.
        correct.append(test_acc)
    print('++++++++++++++++++++++++ ONLY TABLE ++++++++++++++++++++++++++++')
    print(classification_report(gt_rec.type(torch.int64), pred_rec))
    print(confusion_matrix(gt_rec.type(torch.int64), pred_rec))
    return np.mean(correct)

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

