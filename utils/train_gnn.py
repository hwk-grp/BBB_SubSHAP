import torch
import numpy
from sklearn.metrics import classification_report, roc_auc_score, f1_score, accuracy_score
from utils.util import softmax


def train(model, data_loader, criterion, optimizer, scheduler):
    model.train()
    train_loss = 0

    preds_list = []
    targets_list = []

    for i, batch in enumerate(data_loader):

        batch = batch.cuda()
        preds = model(batch)

        preds = preds.squeeze()
        batch.y = batch.y.squeeze()

        loss = criterion(preds, batch.y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.detach().item()

        preds = preds.cpu().detach().numpy()
        batch.y = batch.y.cpu().numpy()
        preds_list.append(preds)
        targets_list.append(batch.y)

    # learning rate scheduling
    scheduler.step()

    preds_list_all = numpy.concatenate(preds_list)
    targets_list_all = numpy.concatenate(targets_list)

    preds_list_all_softmax = softmax(preds_list_all)
    preds_list_all_np = numpy.argmax(preds_list_all_softmax, axis=1)

    # Performance matrices
    train_acc = accuracy_score(targets_list_all, preds_list_all_np)
    train_f1 = f1_score(targets_list_all, preds_list_all_np)
    train_auc = roc_auc_score(targets_list_all, preds_list_all_np)
    train_p_auc = roc_auc_score(targets_list_all, preds_list_all_softmax[:, 1])

    return train_loss / len(data_loader), train_acc, train_f1, train_auc, train_p_auc


def test(model, data_loader, criterion):
    model.eval()

    preds_list = []
    targets_list = []

    with torch.no_grad():
        test_loss = 0

        for i, batch in enumerate(data_loader):

            batch = batch.cuda()
            preds = model(batch)

            preds = preds.squeeze()
            batch.y = batch.y.squeeze()

            loss = criterion(preds, batch.y)
            test_loss += loss.detach().item()

            preds = preds.cpu().numpy()
            batch.y = batch.y.cpu().numpy()
            preds_list.append(preds)
            targets_list.append(batch.y)


        preds_list_all = numpy.concatenate(preds_list)
        targets_list_all = numpy.concatenate(targets_list)

        preds_list_all_softmax = softmax(preds_list_all)
        preds_list_all_np = numpy.argmax(preds_list_all_softmax, axis=1)

        # Performance matrices
        test_acc = accuracy_score(targets_list_all, preds_list_all_np)
        test_f1 = f1_score(targets_list_all, preds_list_all_np)
        test_auc = roc_auc_score(targets_list_all, preds_list_all_np)
        test_p_auc = roc_auc_score(targets_list_all, preds_list_all_softmax[:, 1])

    return test_loss / len(data_loader), test_acc, test_f1, test_auc, test_p_auc
