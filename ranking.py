import numpy as np
import pandas as pd
import time
import re
import math
import random
import pickle
import os
import model as mymodel

from sklearn.model_selection import train_test_split
from sklearn import metrics

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.modules import Module
from torch.utils.data import Dataset, DataLoader

from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import train_test_split_edges
from torch_geometric.utils import add_remaining_self_loops, add_self_loops
from torch_geometric.utils import to_undirected
from torch_geometric.nn import GCNConv, SAGEConv,GAE, VGAE

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score


def baseline(X_train, y_train, y_test, X_test, le, types, top_items_idx, topk, trials_drug, indices_train):

    clf = LogisticRegression().fit(X_train, y_train)
    print("Logit AUROC", roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))
    print("Logit AUPRC", average_precision_score(y_test, clf.predict_proba(X_test)[:, 1]))

    clf = GradientBoostingClassifier().fit(X_train, y_train)
    print("XGBoost AUROC", roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))
    print("XGBoost AUPRC", average_precision_score(y_test, clf.predict_proba(X_test)[:, 1]))

    clf = RandomForestClassifier().fit(X_train, y_train)
    print("rf AUROC", roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))
    print("rf AUPRC", average_precision_score(y_test, clf.predict_proba(X_test)[:, 1]))

    clf = make_pipeline(StandardScaler(), SVC(gamma='auto', probability=True)).fit(X_train, y_train)
    print("svm AUROC", roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))
    print("svm AUPRC", average_precision_score(y_test, clf.predict_proba(X_test)[:, 1]))

    topk_drugs = pd.DataFrame([(rank, drug.split('_')[1]) for rank, drug in enumerate(
        le.inverse_transform((types == 'drug').nonzero()[0][top_items_idx])[:topk + 1])], columns=['rank', 'drug'])
    topk_drugs['under_trials'] = topk_drugs['drug'].isin(trials_drug).astype(int)
    topk_drugs['is_used_in_training'] = topk_drugs['drug'].isin(
        np.array([drug.split('_')[1] for drug in le.classes_[types == 'drug']])[indices_train]).astype(int)
    topk_drugs.to_csv('top300_drugs.csv')

def train_test_split_edges(data, val_ratio=0.05, test_ratio=0.1):
    r"""Splits the edges of a :obj:`torch_geometric.data.Data` object
    into positive and negative train/val/test edges, and adds attributes of
    `train_pos_edge_index`, `train_neg_adj_mask`, `val_pos_edge_index`,
    `val_neg_edge_index`, `test_pos_edge_index`, and `test_neg_edge_index`
    to :attr:`data`.

    Args:
        data (Data): The data object.
        val_ratio (float, optional): The ratio of positive validation
            edges. (default: :obj:`0.05`)
        test_ratio (float, optional): The ratio of positive test
            edges. (default: :obj:`0.1`)

    :rtype: :class:`torch_geometric.data.Data`
    """

    assert 'batch' not in data  # No batch-mode.

    num_nodes = data.num_nodes
    row, col = data.edge_index
    # data.edge_index = None
    attr = data.edge_attr

    # Return upper triangular portion.
    # mask = row < col
    # row, col = row[mask], col[mask]

    n_v = int(math.floor(val_ratio * row.size(0)))
    n_t = int(math.floor(test_ratio * row.size(0)))

    # Positive edges.
    perm = torch.randperm(row.size(0))
    row, col = row[perm], col[perm]
    attr = attr[perm]

    r, c = row[:n_v], col[:n_v]
    data.val_pos_edge_index = torch.stack([r, c], dim=0)
    data.val_pos_edge_attr = attr[:n_v]

    r, c = row[n_v:n_v + n_t], col[n_v:n_v + n_t]
    data.test_pos_edge_index = torch.stack([r, c], dim=0)
    data.test_post_edge_attr = attr[n_v:n_v + n_t]

    r, c = row[n_v + n_t:], col[n_v + n_t:]
    data.train_pos_edge_index = torch.stack([r, c], dim=0)
    data.train_pos_edge_attr = attr[n_v + n_t:]

    # Negative edges.
    neg_adj_mask = torch.ones(num_nodes, num_nodes, dtype=torch.uint8)
    neg_adj_mask = neg_adj_mask.triu(diagonal=1).to(torch.bool)
    neg_adj_mask[row, col] = 0

    neg_row, neg_col = neg_adj_mask.nonzero().t()
    perm = random.sample(range(neg_row.size(0)),
                         min(n_v + n_t, neg_row.size(0)))
    perm = torch.tensor(perm)
    perm = perm.to(torch.long)
    neg_row, neg_col = neg_row[perm], neg_col[perm]

    neg_adj_mask[neg_row, neg_col] = 0
    data.train_neg_adj_mask = neg_adj_mask

    row, col = neg_row[:n_v], neg_col[:n_v]
    data.val_neg_edge_index = torch.stack([row, col], dim=0)

    row, col = neg_row[n_v:n_v + n_t], neg_col[n_v:n_v + n_t]
    data.test_neg_edge_index = torch.stack([row, col], dim=0)

    return data

def main():
    data_path = "data_processed/"
    USE_CUDA = False
    exp_id = 'v0'
    NUM_EPOCHS = 10
    NUM_WORKERS = os.cpu_count()
    embedding_size = 128

    device = torch.device("cuda" if USE_CUDA and torch.cuda.is_available() else "cpu")
    torch.manual_seed(1)
    if device.type == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    le = pickle.load(open(data_path + 'LabelEncoder_' + exp_id + '.pkl', 'rb'))
    edge_index = pickle.load(open(data_path + 'edge_index_' + exp_id + '.pkl', 'rb'))
    node_feature_np = pickle.load(open(data_path + 'node_feature_' + exp_id + '.pkl', 'rb'))

    node_feature = torch.tensor(node_feature_np, dtype=torch.float)
    edge = torch.tensor(edge_index[['node1', 'node2']].values, dtype=torch.long)
    edge_attr_dict = {'gene-drug': 0, 'gene-gene': 1, 'bait-gene': 2, 'gene-phenotype': 3, 'drug-phenotype': 4}
    edge_index['type'] = edge_index['type'].apply(lambda x: edge_attr_dict[x])

    # print(edge_index['type'].value_counts())
    edge_attr = torch.tensor(edge_index['type'].values, dtype=torch.long)
    data = Data(x=node_feature,edge_index=edge.t().contiguous(),edge_attr=edge_attr)
    print(data.num_features, data.num_nodes, data.num_edges)
    # print(edge_attr.size())
    # print(data.contains_isolated_nodes(), data.is_directed())

    data_split=train_test_split_edges(data, test_ratio=0.1, val_ratio=0)
    # print(data_split)
    x,train_pos_edge_index,train_pos_edge_attr = data_split.x.to(device), data_split.train_pos_edge_index.to(device), data_split.train_pos_edge_attr.to(device)

    train_pos_edge_index, train_pos_edge_attr = add_remaining_self_loops(train_pos_edge_index, train_pos_edge_attr)

    print(pd.Series(train_pos_edge_attr.cpu().numpy()).value_counts())

    x, train_pos_edge_index, train_pos_edge_attr = Variable(x), Variable(train_pos_edge_index), Variable(train_pos_edge_attr)

    model = VGAE(mymodel.Encoder_VGAE(node_feature.shape[1], embedding_size)).to(device)
    optimizer = torch.optim.Adam(model.parameters())

    def train():
        model.train()
        optimizer.zero_grad()
        z = model.encode(x, train_pos_edge_index, train_pos_edge_attr)
        loss = model.recon_loss(z, train_pos_edge_index)
        loss = loss + (1 / data.num_nodes) * model.kl_loss()
        loss.backward()
        optimizer.step()
        print(loss.item())

    def test(pos_edge_index, neg_edge_index):
        model.eval()
        with torch.no_grad():
            z = model.encode(x, train_pos_edge_index, train_pos_edge_attr)
        return model.test(z, pos_edge_index, neg_edge_index)

    model.test(x, data_split.test_pos_edge_index, data_split.test_neg_edge_index)

    for epoch in range(1,NUM_EPOCHS+1):
        train()
        auc, ap = test(data_split.test_pos_edge_index, data_split.test_neg_edge_index)
        print('Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, auc, ap))

    model.eval()
    z = model.encode(x, data.edge_index.to(device), data.edge_attr.to(device))
    z_np = z.squeeze().detach().cpu().numpy()

    pickle.dump(z_np, open(data_path + 'node_embedding_' + exp_id + '.pkl', 'wb'))
    torch.save(model.state_dict(), data_path+'VAE_encoders_'+exp_id+'.pkl')

    model.load_state_dict(torch.load(data_path + 'VAE_encoders_' + exp_id + '.pkl'))
    model.eval()

    topk = 300
    types = np.array([item.split('_')[0] for item in le.classes_])
    # label
    trials = pd.read_excel('data_raw/All_trails_5_24.xlsx', header=1, index_col=0)
    trials_drug = set([drug.strip().upper() for lst in
                       trials.loc[trials['study_category'].apply(lambda x: 'drug' in x.lower()), 'intervention'].apply(
                           lambda x: re.split(r'[+|/|,]', x.replace(' vs. ', '/').replace(' vs ', '/').replace(' or ', '/').replace(
                                    ' with and without ', '/').replace(' /wo ', '/').replace(' /w ', '/').replace(
                                    ' and ', '/').replace(' - ', '/').replace(' (', '/').replace(') ','/'))).values for drug in lst])
    drug_labels = [1 if drug.split('_')[1] in trials_drug else 0 for drug in le.classes_[types == 'drug']]

    seed = 70
    indices = np.arange(len(drug_labels))
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(z_np[types == 'drug'], drug_labels,
                                                                                     indices, test_size=0.5,
                                                                                     random_state=seed, )
    _X_train, _y_train = Variable(torch.tensor(X_train, dtype=torch.float).to(device)), Variable(
        torch.tensor(y_train, dtype=torch.float).to(device))
    _X_test, _y_test = Variable(torch.tensor(X_test, dtype=torch.float).to(device)), Variable(
        torch.tensor(y_test, dtype=torch.float).to(device))

    clf = mymodel.Classifier(embedding_size).to(device)
    optimizer = torch.optim.Adam(clf.parameters())
    criterion = mymodel.BPRLoss(num_neg_samples=15)

    best_auprc = 0
    for epoch in range(30):
        clf.train()
        optimizer.zero_grad()
        out = clf(_X_train)
        loss = criterion(out.squeeze(), _y_train)
        loss.backward()
        optimizer.step()
        print('training loss', loss.item())

        clf.eval()
        print('test loss', criterion(clf(_X_test).squeeze(), _y_test).item())
        prob = torch.sigmoid(clf(_X_test)).cpu().detach().numpy().squeeze()
        auprc = metrics.average_precision_score(y_test, prob)
        if auprc > best_auprc:
            best_auproc = auprc
            torch.save(clf, data_path + 'nn_clf.pt')

    clf.load_state_dict(torch.load(data_path + 'nn_clf.pt').state_dict())

    clf.eval()

    prob = torch.sigmoid(clf(_X_test)).cpu().detach().numpy().squeeze()
    print("AUROC", metrics.roc_auc_score(y_test, prob))
    print("AUPRC", metrics.average_precision_score(y_test, prob))

    top_items_idx = np.argsort(-clf(torch.tensor(z_np[types == 'drug'], dtype=torch.float).to(device)).squeeze().detach().cpu().numpy())

    baseline(X_train, y_train, y_test, X_test, le, types, top_items_idx, topk, trials_drug, indices_train)

if __name__ == "__main__":
    main()