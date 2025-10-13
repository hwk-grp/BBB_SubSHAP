import torch
import numpy
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, confusion_matrix
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

    tn, fp, fn, tp = confusion_matrix(targets_list_all, preds_list_all_np).ravel()
    train_sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0.0  # True positive rate (recall)
    train_specificity = tn / (tn + fp) if (tn + fp) != 0 else 0.0  # True negative rate
    train_fpr_value = fp / (fp + tn) if (fp + tn) != 0 else 0.0  # False positive rate

    return (train_loss / len(data_loader), train_acc, train_f1, train_auc, train_p_auc, train_sensitivity, train_specificity, train_fpr_value)


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

        tn, fp, fn, tp = confusion_matrix(targets_list_all, preds_list_all_np).ravel()
        test_sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0.0  # True positive rate (recall)
        test_specificity = tn / (tn + fp) if (tn + fp) != 0 else 0.0  # True negative rate
        test_fpr_value = fp / (fp + tn) if (fp + tn) != 0 else 0.0  # False positive rate

    return (test_loss / len(data_loader), test_acc, test_f1, test_auc, test_p_auc, test_sensitivity, test_specificity, test_fpr_value)
