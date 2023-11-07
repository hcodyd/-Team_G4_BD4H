import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import numpy as np
from torch.utils.data import BatchSampler, WeightedRandomSampler
from torch_geometric.nn import GCNConv, SAGEConv,GAE, VGAE

class Encoder_VGAE(nn.Module):
    def __init__(self, in_channels, out_channels, isClassificationTask=False):
        super(Encoder_VGAE, self).__init__()
        self.isClassificationTask = isClassificationTask
        self.conv_gene_drug = SAGEConv(in_channels, 2 * out_channels, )
        self.conv_gene_gene = SAGEConv(in_channels, 2 * out_channels, )
        self.conv_bait_gene = SAGEConv(in_channels, 2 * out_channels, )
        self.conv_gene_phenotype = SAGEConv(in_channels, 2 * out_channels, )
        self.conv_drug_phenotype = SAGEConv(in_channels, 2 * out_channels)

        self.bn = nn.BatchNorm1d(5 * 2 * out_channels)
        # variational encoder
        self.conv_mu = SAGEConv(5 * 2 * out_channels, out_channels, )
        self.conv_logvar = SAGEConv(5 * 2 * out_channels, out_channels, )

    def forward(self, x, edge_index, edge_attr):
        x = F.dropout(x, training=self.training)

        index_gene_drug = (edge_attr == 0).nonzero().reshape(1, -1)[0]
        edge_index_gene_drug = edge_index[:, index_gene_drug]

        index_gene_gene = (edge_attr == 1).nonzero().reshape(1, -1)[0]
        edge_index_gene_gene = edge_index[:, index_gene_gene]

        index_bait_gene = (edge_attr == 2).nonzero().reshape(1, -1)[0]
        edge_index_bait_gene = edge_index[:, index_bait_gene]

        index_gene_phenotype = (edge_attr == 3).nonzero().reshape(1, -1)[0]
        edge_index_gene_phenotype = edge_index[:, index_gene_phenotype]

        index_drug_phenotype = (edge_attr == 4).nonzero().reshape(1, -1)[0]
        edge_index_drug_phenotype = edge_index[:, index_drug_phenotype]

        x_gene_drug = F.dropout(F.relu(self.conv_gene_drug(x, edge_index_gene_drug)), p=0.5, training=self.training, )
        x_gene_gene = F.dropout(F.relu(self.conv_gene_gene(x, edge_index_gene_gene)), p=0.5, training=self.training)
        x_bait_gene = F.dropout(F.relu(self.conv_bait_gene(x, edge_index_bait_gene)), p=0.1, training=self.training)
        x_gene_phenotype = F.dropout(F.relu(self.conv_gene_phenotype(x, edge_index_gene_phenotype)),
                                     training=self.training)
        x_drug_phenotype = F.dropout(F.relu(self.conv_drug_phenotype(x, edge_index_drug_phenotype)),
                                     training=self.training)

        x = self.bn(torch.cat([x_gene_drug, x_gene_gene, x_bait_gene, x_gene_phenotype, x_drug_phenotype], dim=1))

        return self.conv_mu(x, edge_index), self.conv_logvar(x, edge_index)

class Classifier(nn.Module):
    def __init__(self,embedding_dim):
        super(Classifier, self).__init__()
        self.fc1=nn.Linear(embedding_dim,embedding_dim)
        self.fc2=nn.Linear(embedding_dim,1)
        self.bn=nn.BatchNorm1d(embedding_dim)
    def forward(self, x):
        residual1 = x
        x = F.dropout(x, training=self.training)
        x= self.bn(F.dropout(F.relu(self.fc1(x)),training=self.training))
        x += residual1
        return self.fc2(x)


class BPRLoss(nn.Module):
    def __init__(self, num_neg_samples):
        super(BPRLoss, self).__init__()
        self.num_neg_samples = num_neg_samples

    def forward(self, output, label):
        positive_output = output[label == 1]
        negative_output = output[label != 1]

        # negative sample proportional to the high values
        negative_sampler = WeightedRandomSampler(negative_output - min(negative_output),
                                                 num_samples=self.num_neg_samples * len(positive_output),
                                                 replacement=True)
        negative_sample_output = negative_output[
            torch.tensor(list(BatchSampler(negative_sampler, batch_size=len(positive_output), drop_last=True)),
                         dtype=torch.long).t()]
        return -(positive_output.view(-1, 1) - negative_sample_output).sigmoid().log().mean()