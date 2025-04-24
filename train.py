import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from matplotlib import pyplot as plt
import copy
import time
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, recall_score, \
    precision_score, precision_recall_curve, auc
import os
import random
import pandas as pd
from torch.autograd import Variable
import csv
import pickle
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.nn import CrossEntropyLoss

def set_random_seed(seed, deterministic=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


set_random_seed(2024, deterministic=True)


def gen_ran_output(data, model, args, output, label, device):

    model.zero_grad()
    criterion = torch.nn.CrossEntropyLoss()
    temp_loss = criterion(output, label.long())
    temp_loss.backward(retain_graph=True)

    vice_model = copy.deepcopy(model)


    for (name, vice_param), (name, param) in zip(vice_model.named_parameters(), model.named_parameters()):
        if name.split('.')[0] == 'gnn':
            if len(param.data) == 1:
                vice_param.data = param.data
            else:
                alpha = args.alpha
                grad_direction = param.grad.sign() if param.grad is not None else torch.zeros_like(param.data)
                random_noise = torch.normal(0, torch.ones_like(param.data) * param.data.std()).to(device)
                loss_weight = temp_loss.item() / (temp_loss.item() + 1e-6)
                vice_param.data = param.data + loss_weight * args.eta * (
                    alpha * grad_direction + (1 - alpha) * random_noise
                )
        else:
            vice_param.data = param.data

    model.zero_grad()

    with torch.no_grad():
        _, _, _, z2_o, _, z2_sim = vice_model.forward_cl(data)
    return z2_o.detach(), z2_sim.detach()


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean', num_classes=100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is None:
            self.alpha = torch.ones(num_classes).cuda()
        else:
            self.alpha = alpha.cuda()

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha[targets] * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class NormalizedCrossEntropyLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, logits):
        logits = logits / self.temperature
        labels = torch.zeros(logits.size(0), dtype=torch.long).to(logits.device)
        return self.cross_entropy(logits, labels)


def train_model(model, optimizer, data_o, data_o_sim, data_s, data_s_sim, data_a, data_a_sim, train_loader, val_loader, test_loader, args, train_data):
    m = torch.nn.Sigmoid()
    # drugbank
    if hasattr(args, 'use_class_weights') and args.use_class_weights:
        label_counts = torch.zeros(100)
        for _, (inp) in enumerate(train_loader):
            labels = inp[2]
            for label in labels:
                label_counts[label] += 1
        total_samples = label_counts.sum()
        class_weights = total_samples / (len(label_counts) * label_counts)
        class_weights = class_weights / class_weights.sum()
    else:
        class_weights = None
    loss_fct = FocalLoss(alpha=class_weights, gamma=2)

    criterion = LabelSmoothingLoss(classes=3, smoothing=0.1)

    b_xent = nn.BCEWithLogitsLoss()
    nce_criterion = NormalizedCrossEntropyLoss(temperature=args.tau)
    loss_history = []
    max_auc = 0
    max_f1 = 0
    vaild_data = args.vaild_data

    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=5,
        T_mult=2,
        eta_min=1e-6
    )


    if args.cuda:
        model.to('cuda')
        data_o.to('cuda')
        data_s.to('cuda')
        data_a.to('cuda')
        data_o_sim.to('cuda')
        data_s_sim.to('cuda')
        data_a_sim.to('cuda')
        train_data.to('cuda')
        vaild_data.to('cuda')

    lbl = data_a.y
    t_total = time.time()
    model_max = copy.deepcopy(model)
    print('Start Training...')
    stoping = 0
    all_time = []

    for epoch in range(args.epochs):
        t = time.time()
        print('-------- Epoch ' + str(epoch + 1) + ' --------')
        y_pred_train = []
        y_label_train = []

        for i, (inp) in enumerate(train_loader):
            label = inp[2]
            label = np.array(label, dtype=np.int64)
            label = torch.from_numpy(label)
            if args.cuda:
                label = label.cuda()

            model.train()
            optimizer.zero_grad()

            x, x_sim, x1_o, x2_o, x1_sim, x2_sim = model.forward_cl(train_data)
            output = model.forward_classification(x1_o, x2_o, x1_sim, x2_sim, inp)
            output = torch.squeeze(output)
            z2_o, z2_sim = gen_ran_output(train_data, model, args, output, label, device='cuda')
            z2_o = Variable(z2_o.detach().data, requires_grad=False)
            z2_sim = Variable(z2_sim.detach().data, requires_grad=False)

           
            # loss1 = loss_fct(output, label.long())  # focal loss(drugbank)
            loss1 = criterion(output, label) #(ddinter)


            x1_x2_org = model.loss_cl(x, x2_o, z2_o)
            loss2_org = nce_criterion(x1_x2_org)
            x1_x2_sim = model.loss_cl(x_sim, x2_sim, z2_sim)
            loss2_sim = nce_criterion(x1_x2_sim)
            loss2 = loss2_org + loss2_sim


            loss_train = args.loss_ratio1 * loss1 + args.loss_ratio3  * loss2
            loss_history.append(loss_train.cpu().detach().numpy())
            loss_train.backward()
            optimizer.step()


            label_ids = label.to('cpu').numpy()
            y_label_train = y_label_train + label_ids.flatten().tolist()
            y_pred_train = y_pred_train + output.flatten().tolist()

            if i % 100 == 0:
                print('epoch: ' + str(epoch + 1) +
                      '/ iteration: ' + str(i + 1) +
                      '/ loss_train: ' + str(loss_train.cpu().detach().numpy()))
        scheduler.step()
        end_time = time.time() - t
        all_time.append(end_time)

        y_pred_train1 = []
        y_label_train = np.array(y_label_train)

        y_pred_train = np.array(y_pred_train).reshape((-1,3))
        # y_pred_train = np.array(y_pred_train).reshape((-1, 100))
        for i in range(y_pred_train.shape[0]):
            a = np.max(y_pred_train[i])
            for j in range(y_pred_train.shape[1]):
                if y_pred_train[i][j] == a:
                    y_pred_train1.append(j)
                    break

        acc = accuracy_score(y_label_train, y_pred_train1)
        f1_score1 = f1_score(y_label_train, y_pred_train1, average='macro')
        recall1 = recall_score(y_label_train, y_pred_train1, average='macro')
        precision1 = precision_score(y_label_train, y_pred_train1, average='macro')

        drug_list = []
        with open(f'trimnet/data/{args.dataset}/{args.dataset}_smiles.csv', 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                drug_list.append(row[0])

        if not args.fastmode:
            acc_val, f1_val, recall_val, precision_val, loss_val, _, _, _, _ = test(
                model, val_loader, data_o, data_o_sim, data_s, data_s_sim, data_a, data_a_sim, args, 0)

            if acc_val >= max_auc and f1_val >= max_f1:
                model_max = copy.deepcopy(model)
                max_auc = acc_val
                max_f1 = f1_val
                stoping = 0
                print(f"best model is {epoch}")
            else:
                stoping += 1

            print('epoch: {:04d}'.format(epoch + 1),
                  'loss_train: {:.4f}'.format(loss_train.item()),
                  'auroc_train: {:.4f}'.format(acc),
                  'loss_val: {:.4f}'.format(loss_val.item()),
                  'acc_val: {:.4f}'.format(acc_val),
                  'f1_val: {:.4f}'.format(f1_val),
                  'recall_val: {:.4f}'.format(recall_val),
                  'precision_val: {:.4f}'.format(precision_val),
                  'time: {:.4f}s'.format(time.time() - t))
        else:
            model_max = copy.deepcopy(model)

        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
    df = pd.DataFrame({'Epoch': list(range(1, args.epochs + 1)), 'Time(seconds)': all_time})
    df.to_csv('epoch_time_{}.csv'.format(args.zhongzi), index=False)
    print("epoch times saved")
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    acc_test, f1_test, recall_test, precision_test, loss_test, save_true_label, save_pred_label, save_class_label, output_list = test(
        model_max, test_loader, data_o, data_o_sim, data_s, data_s_sim, data_a, data_a_sim, args, 1)

    def writelabel(filename, pred_labels, true_labels, class_label):
        file = open(filename, 'w')
        for i in range(len(pred_labels)):
            file.write(str(pred_labels[i]) + ' ' + str(true_labels[i]) + ' ' + str(class_label[i]) + '\n')

    output_file = 'true_pred_label_addreadout{}.txt'.format(args.zhongzi)
    writelabel(output_file, save_true_label, save_pred_label, save_class_label)
    np.save('output_embedding_addreadout{}.npy'.format(args.zhongzi), output_list)

    print('loss_test: {:.4f}'.format(loss_test.item()), 'acc_test: {:.4f}'.format(acc_test),
          'f1_test: {:.4f}'.format(f1_test), 'precision_test: {:.4f}'.format(precision_test),
          'recall_test: {:.4f}'.format(recall_test))


def test(model, loader, data_o, data_o_sim, data_s, data_s_sim, data_a, data_a_sim, args, printfou):
    m = torch.nn.Sigmoid()
    loss_fct = torch.nn.CrossEntropyLoss()
    b_xent = nn.BCEWithLogitsLoss()
    model.eval()
    y_pred = []
    y_label = []
    y_class_label = []
    output_list = []
    lbl = data_a.y
    test_data = args.train_data
    zhongzi = args.zhongzi

    with torch.no_grad():
        for i, (inp) in enumerate(loader):
            label = inp[2]
            label = np.array(label, dtype=np.int64)
            label = torch.from_numpy(label)
            class_label = inp[3]
            class_label = np.array(class_label, dtype=np.int64)
            class_label = torch.from_numpy(class_label)
            if args.cuda:
                label = label.cuda()
                class_label = class_label.cuda()
                test_data = test_data.cuda()
            x, x_sim, x1_o, x2_o, x1_sim, x2_sim = model.forward_cl(test_data)
            output = model.forward_classification(x1_o, x2_o, x1_sim, x2_sim, inp)
            log = torch.squeeze(m(output))

            loss1 = loss_fct(log, label.long())
            loss  = args.loss_ratio1 * loss1

            label_ids = label.to('cpu').numpy()
            class_label_ids = class_label.to('cpu').numpy()
            output_list.append(output.cpu().numpy())
            y_label = y_label + label_ids.flatten().tolist()
            y_class_label = y_class_label + class_label_ids.flatten().tolist()
            y_pred = y_pred + output.flatten().tolist()
    output_list = np.concatenate(output_list)
    y_pred_train1 = []
    y_label_train = np.array(y_label)
    y_class_label_train = np.array(y_class_label)

    y_pred_train = np.array(y_pred)
    y_pred_train = y_pred_train.reshape((-1, 3))
    # y_pred_train = y_pred_train.reshape((-1, 100))
    for i in range(y_pred_train.shape[0]):
        a = np.max(y_pred_train[i])
        for j in range(y_pred_train.shape[1]):
            if y_pred_train[i][j] == a:
                y_pred_train1.append(j)
                break
    save_true_label, save_pred_label, save_class_label = y_label_train, y_pred_train1, y_class_label_train
    acc = accuracy_score(y_label_train, y_pred_train1)
    f1_score1 = f1_score(y_label_train, y_pred_train1, average='macro')
    recall1 = recall_score(y_label_train, y_pred_train1, average='macro')
    precision1 = precision_score(y_label_train, y_pred_train1, average='macro')
    y_label_train1 = np.zeros((y_label_train.shape[0], 3))
    # y_label_train1 = np.zeros((y_label_train.shape[0], 100))
    for i in range(y_label_train.shape[0]):
        y_label_train1[i][y_label_train[i]] = 1

    auc_hong = 0
    aupr_hong = 0
    nn1 = y_label_train1.shape[1]
    for i in range(y_label_train1.shape[1]):
        if np.sum(y_label_train1[:, i].reshape((-1))) < 1:
            nn1 = nn1 - 1
            continue
        else:
            auc_hong = auc_hong + roc_auc_score(y_label_train1[:, i].reshape((-1)), y_pred_train[:, i].reshape((-1)))
            precision, recall, _thresholds = precision_recall_curve(y_label_train1[:, i].reshape((-1)),
                                                                    y_pred_train[:, i].reshape((-1)))
            aupr_hong = aupr_hong + auc(recall, precision)

    auc_macro = auc_hong / nn1
    aupr_macro = aupr_hong / nn1
    auc1 = roc_auc_score(y_label_train1.reshape((-1)), y_pred_train.reshape((-1)), average='micro')
    precision, recall, _thresholds = precision_recall_curve(y_label_train1.reshape((-1)), y_pred_train.reshape((-1)))
    aupr = auc(recall, precision)

    if printfou == 1:
        with open(args.out_file, 'a') as f:
            f.write(str(zhongzi) + '  ' + str(acc) + '   ' + str(f1_score1) + '   ' + str(recall1) + '   ' + str(
                precision1) + '   ' + str(auc1) + '   ' + str(aupr) + '   ' + str(auc_macro) + '   ' + str(
                aupr_macro) + '\n')

    return acc, f1_score1, recall1, precision1, loss, save_true_label, save_pred_label, save_class_label, output_list
