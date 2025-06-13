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

        train_x = batch[:, :-1].clone().detach().float()
        train_y = batch[:, -1].clone().detach().long().view(-1, 1)

        train_x = train_x.cuda()
        train_y = train_y.cuda()

        preds = model(train_x)

        preds = preds.squeeze()
        train_y = train_y.squeeze()

        loss = criterion(preds, train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.detach().item()

        preds = preds.cpu().detach().numpy()
        train_y = train_y.cpu().numpy()
        preds_list.append(preds)
        targets_list.append(train_y)

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

            train_x = batch[:, :-1].clone().detach().float()
            train_y = batch[:, -1].clone().detach().long().view(-1, 1)

            train_x = train_x.cuda()
            train_y = train_y.cuda()

            preds = model(train_x)

            preds = preds.squeeze()
            train_y = train_y.squeeze()

            loss = criterion(preds, train_y)
            test_loss += loss.detach().item()

            preds = preds.cpu().numpy()
            train_y = train_y.cpu().numpy()
            preds_list.append(preds)
            targets_list.append(train_y)

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
