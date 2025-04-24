import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch_geometric.nn import HGTConv, GCNConv
from torch_scatter import scatter_mean
from torch_geometric.nn import global_mean_pool
import numpy as np
import csv
import random
import pickle


def set_random_seed(seed, deterministic=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


set_random_seed(1, deterministic=True)


def reset_parameters(w):
    stdv = 1. / math.sqrt(w.size(0))
    w.data.uniform_(-stdv, stdv)


class Discriminator(nn.Module):
    def __init__(self, n_h, temperature=0.1):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(32, 32, 1)
        self.f_k_sim = nn.Bilinear(64, 64, 1)
        self.temperature = temperature

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = c.expand_as(h_pl)
        sc_1 = self.f_k(h_pl, c_x)
        sc_2 = self.f_k(h_mi, c_x)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1) / self.temperature
        return logits

class HierarchicalFeatureFusion(nn.Module):
    def __init__(self, trim_dim=128, gcn_dim=128, con_dim=96, sim_dim=128, hidden_dim=512):
        super().__init__()

        self.dropout = nn.Dropout(0.3)

        self.structure_residual = nn.Linear(trim_dim + gcn_dim, hidden_dim)
        self.structure_fusion = nn.Sequential(
            nn.Linear(trim_dim + gcn_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            self.dropout
        )

        self.contrastive_residual = nn.Linear(con_dim * 2, hidden_dim)
        self.contrastive_fusion = nn.Sequential(
            nn.Linear(con_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            self.dropout
        )

        self.cross_attention = CrossModalAttention(hidden_dim)

        self.final_residual = nn.Linear(hidden_dim * 3, hidden_dim)
        self.final_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            self.dropout
        )

        self.feature_rescale = nn.Parameter(torch.ones(1, hidden_dim))
        self.feature_shift = nn.Parameter(torch.zeros(1, hidden_dim))

    def forward(self, trimnet, tri_gcn, trim_con, similarity, sim_con):
        structure_input = torch.cat([trimnet, tri_gcn], dim=-1)
        structure_residual = self.structure_residual(structure_input)
        structure_feat = self.structure_fusion(structure_input) + structure_residual

        contrastive_input = torch.cat([trim_con, sim_con], dim=-1)
        contrastive_residual = self.contrastive_residual(contrastive_input)
        contrastive_feat = self.contrastive_fusion(contrastive_input) + contrastive_residual

        fused_features = self.cross_attention(
            structure_feat,
            contrastive_feat,
            similarity
        )

        final_residual = self.final_residual(fused_features)
        output = self.final_fusion(fused_features) + final_residual

        output = output * self.feature_rescale + self.feature_shift

        return output


class CrossModalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.2,
            batch_first=True
        )

        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

    def forward(self, structure_feat, contrastive_feat, similarity_feat):
        stacked_features = torch.stack(
            [structure_feat, contrastive_feat, similarity_feat],
            dim=1
        )

        normed_features = self.layer_norm1(stacked_features)
        attended_features, _ = self.attention(
            normed_features,
            normed_features,
            normed_features
        )
        attended_features = attended_features + stacked_features

        normed_features = self.layer_norm2(attended_features)
        output = self.feed_forward(normed_features)
        output = output + attended_features

        output = output.reshape(output.size(0), -1)
        return output


class InteractionEmbedding(nn.Module):
    def __init__(self, n_drug1, n_drug2, embedding_dim, dropout=0.2):
        super().__init__()
        self.drug_project1 = nn.Linear(n_drug1, embedding_dim, bias=False)
        self.drug_project2 = nn.Linear(n_drug2, embedding_dim, bias=False)
        self.alpha = nn.Parameter(torch.tensor(0.8))
        self.dropout = nn.Dropout(dropout)
        self.output_dim = embedding_dim

    def _process_embedding(self, drug_embedding, project_layer):
        identity = torch.eye(drug_embedding.size(0), device=drug_embedding.device)
        learned = project_layer(identity)
        projected = project_layer(drug_embedding.float())
        return self.alpha * learned + (1 - self.alpha) * projected

    def forward(self, association_pairs, drug_embedding1, drug_embedding2):
        drug_embedding1 = self._process_embedding(drug_embedding1, self.drug_project1)
        drug_embedding2 = self._process_embedding(drug_embedding2, self.drug_project2)

        emb1 = drug_embedding1[association_pairs[0]]
        emb2 = drug_embedding2[association_pairs[1]]

        associations = F.normalize(emb1 * emb2, p=2, dim=1)
        return self.dropout(associations)


class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq, msk=None):
        if msk is None:
            return torch.mean(seq, 0)
        else:
            msk = torch.unsqueeze(msk, -1)
            return torch.sum(seq * msk, 0) / torch.sum(msk)


class MLP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MLP, self).__init__()

        self.linear1 = nn.Linear(in_channels, 2 * out_channels)
        self.linear2 = nn.Linear(2 * out_channels, out_channels)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)

        return x


class MRCGNN_sub(nn.Module):
    def __init__(self, feature, sim_feature, hidden1, hidden2, decoder1, dropout, zhongzi, train_data):
        super(MRCGNN_sub, self).__init__()

        self.lin = nn.Linear(feature, feature) #(128,128)
        self.lin_sim = nn.Linear(sim_feature, sim_feature) #(1565,1565)
        self.encoder_o1 = HGTConv(feature, hidden1, train_data.metadata(), heads=2) #(128,64)
        self.encoder_o2 = HGTConv(hidden1, hidden2, train_data.metadata(), heads=2) #(64,32)
        self.encoder_o_sim1 = HGTConv(sim_feature, hidden1, train_data.metadata(), heads=2) #(1565,64)
        self.encoder_o_sim2 = HGTConv(hidden1, hidden2, train_data.metadata(), heads=2) #(64,32)
        self.dropout = dropout
        self.sigm = nn.Sigmoid()
        self.read = AvgReadout()

    def forward(self, data):
        # Process original features
        x_dict = {
            node_type: self.lin(x).relu_()
            for node_type, x in data.x_dict.items()
        }
        x1_o = self.encoder_o1(x_dict, data.edge_index_dict) #(1565,128)*(128,64)
        x2_o = self.encoder_o2(x1_o, data.edge_index_dict) #(1565,64)*(64,32)

        # Process similarity features
        x_sim_dict = {
            node_type: self.lin_sim(x).relu_()
            for node_type, x in data.x_sim_dict.items()
        }

        x1_sim = self.encoder_o_sim1(x_sim_dict, data.edge_index_dict)
        x2_sim = self.encoder_o_sim2(x1_sim, data.edge_index_dict)

        x2_os = x2_o['drug']
        x2_os_sim = x2_sim['drug']

        # Readout
        h_os = self.read(x2_os)
        h_os_sim = self.read(x2_os_sim)

        return h_os, h_os_sim, x1_o['drug'], x2_o['drug'], x1_sim['drug'], x2_sim['drug']


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.5):
        super(GCNLayer, self).__init__()
        self.conv = GCNConv(in_features, out_features)
        self.batch_norm = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(p=dropout)

        self.residual = nn.Linear(in_features, out_features) if in_features != out_features else None

    def forward(self, x, edge_index):
        identity = x

        x = self.conv(x, edge_index)
        x = self.batch_norm(x)
        x = F.relu(x)

        if self.residual is not None:
            x = x + self.residual(identity)
        else:
            x = x + identity

        x = self.dropout(x)
        return x


class GCNFeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super(GCNFeatureExtractor, self).__init__()
        self.gcn1 = GCNLayer(input_dim, hidden_dim, dropout)
        self.gcn2 = GCNLayer(hidden_dim, output_dim, dropout)

        self.attention = nn.Sequential(
            nn.Linear(output_dim, 1),
            nn.Tanh()
        )

        self.cached_edge_index = None
        self.edge_types = [(f'drug', str(i), 'drug') for i in range(3-1)]

    def forward(self, x, edge_index_dict):
        if self.cached_edge_index is None:
            edge_indices = []
            for edge_type in self.edge_types:
                if edge_type in edge_index_dict:
                    edge_indices.append(edge_index_dict[edge_type])
            self.cached_edge_index = torch.cat(edge_indices, dim=1)

        x = self.gcn1(x, self.cached_edge_index)
        x = self.gcn2(x, self.cached_edge_index)

        attention_weights = torch.softmax(self.attention(x), dim=0)
        x = x * attention_weights

        return x


class HGCLayer(nn.Module):
    def __init__(self, in_channels, out_channels, metadata, heads=8, dropout=0.5):
        super().__init__()
        self.conv = HGTConv(in_channels, out_channels, metadata, heads=heads)
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        x_dict = {key: self.norm(x) for key, x in x_dict.items()}
        x_dict = {key: self.dropout(x) for key, x in x_dict.items()}
        return x_dict


class HGCFeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, metadata, num_layers=3, heads=8, dropout=0.5):
        super().__init__()
        self.layers = nn.ModuleList()

        self.layers.append(HGCLayer(input_dim, hidden_dim, metadata, heads=heads, dropout=dropout))

        for _ in range(num_layers - 2):
            self.layers.append(HGCLayer(hidden_dim, hidden_dim, metadata, heads=heads, dropout=dropout))

        self.layers.append(HGCLayer(hidden_dim, output_dim, metadata, heads=heads, dropout=dropout))

    def forward(self, x_dict, edge_index_dict):
        for layer in self.layers:
            x_dict = layer(x_dict, edge_index_dict)
        return x_dict


class MRCGNN(nn.Module):
    def __init__(self, feature, sim_feature, hidden1, hidden2, decoder1, dropout, zhongzi, train_data1):
        super(MRCGNN, self).__init__()
        self.gnn = MRCGNN_sub(feature, sim_feature, hidden1, hidden2, decoder1, dropout, zhongzi, train_data1)
        self.gcn_extractor = GCNFeatureExtractor(input_dim=128, hidden_dim=256, output_dim=128, dropout=dropout)
        self.hgc_extractor = HGCFeatureExtractor(
            input_dim=128,
            hidden_dim=128,
            output_dim=128,
            metadata=train_data1.metadata(),
            num_layers=3,
            heads=8,
            dropout=dropout
        )
        self.train_data1 = train_data1

        self.fusion_model = HierarchicalFeatureFusion(
            trim_dim=128,
            gcn_dim=128,
            con_dim=96,
            sim_dim=128,
            hidden_dim=128
        )

        self.interaction_embedding = InteractionEmbedding(
            n_drug1=1565,
            n_drug2=1565,
            embedding_dim=128,
            dropout=dropout
        )
        self.mlp = nn.ModuleList([nn.Linear(128 * 2, 256),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.2),
                                  nn.Linear(256, 128),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.2),
                                  nn.Linear(128, 3)
                                  ])
        self.attt = torch.zeros(2)
        self.attt[0] = 0.5
        self.attt[1] = 0.5
        self.attt = nn.Parameter(self.attt)
        self.disc = Discriminator(hidden2 * 2,temperature=0.1) #(32*2)

        self.projection_head = nn.Sequential(nn.Linear(32, 32), nn.ReLU(inplace=True), nn.Linear(32, 32)) #投影头：用于对比学习
        self.projection_head_sim = nn.Sequential(nn.Linear(32, 32), nn.ReLU(inplace=True), nn.Linear(32, 32))
        drug_list = []
        dataset = 'ddinter'
        with open(f'trimnet/data/{dataset}/{dataset}_smiles.csv', 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                drug_list.append(row[0])
        features = np.load(f'trimnet/data/{dataset}/drug_emb_trimnet' + str(zhongzi) + '.npy')
        ids = np.load(f'trimnet/data/{dataset}/drug_ids.npy')
        ids = ids.tolist()
        features1 = []
        for i in range(len(drug_list)):
            features1.append(features[ids.index(drug_list[i])])
        features1 = np.array(features1)

        with open(f'trimnet/data/{dataset}/drug_similarity.pkl', 'rb') as f:
            sim_data = pickle.load(f)

        sim_drug_ids = sim_data[0]
        sim_features = sim_data[1]

        sim_features1 = []
        for i in range(len(drug_list)):
            sim_features1.append(sim_features[sim_drug_ids.index(drug_list[i])])
        sim_features1 = np.array(sim_features1)


        self.features1 = torch.from_numpy(features1).cuda()
        self.sim_features1 = torch.from_numpy(sim_features1).cuda()

    def MLP(self, vectors, layer):
        for i in range(layer):
            vectors = self.mlp[i](vectors)
        return vectors

    def forward_cl(self, data):
        x, x_sim, x1_o, x2_o, x1_sim, x2_sim = self.gnn(data)
        x = self.projection_head(x)
        x_sim = self.projection_head_sim(x_sim)
        return x, x_sim, x1_o, x2_o, x1_sim, x2_sim

    def forward_classification(self, x1_o, x2_o, x1_sim, x2_sim, idx):
        a = [int(i) for i in list(idx[0])]
        b = [int(i) for i in list(idx[1])]

        aa = torch.tensor(a, dtype=torch.long)
        bb = torch.tensor(b, dtype=torch.long)
        drug_pairs = torch.stack([aa, bb], dim=0).cuda()

        # Layer attention
        final = torch.cat((self.attt[0] * x1_o, self.attt[1] * x2_o),
                          dim=1)
        final_sim = torch.cat((self.attt[0] * x1_sim, self.attt[1] * x2_sim),
                          dim=1)

        entity1_con = final[aa] #(256,96)
        entity2_con = final[bb]
        entity1_con_sim = final_sim[aa]
        entity2_con_sim = final_sim[bb]

        # Skip connection
        # TrimNet
        entity1_tri = self.features1[aa].to(f'cuda')
        entity2_tri = self.features1[bb].to(f'cuda')
        entity1_sim = self.sim_features1[aa].to(f'cuda')
        entity2_sim = self.sim_features1[bb].to(f'cuda')

        edge_index_dict = self.train_data1.edge_index_dict
        gcn_features = self.gcn_extractor(self.features1.to(f'cuda'), edge_index_dict)
        entity1_tri_gcn = gcn_features[aa]
        entity2_tri_gcn = gcn_features[bb]

        sim_features = self.sim_features1.to(f'cuda')

        sim_embedding = self.interaction_embedding(
            drug_pairs,
            sim_features,
            sim_features
        )
        entity1_fused = self.fusion_model(entity1_tri, entity1_tri_gcn, entity1_con, sim_embedding, entity1_con_sim)
        entity2_fused = self.fusion_model(entity2_tri, entity2_tri_gcn, entity2_con, sim_embedding, entity2_con_sim)


        concatenate = torch.cat((entity1_fused.to(torch.float32), entity2_fused.to(torch.float32)), dim=1)
        feature = self.MLP(concatenate, 7)
        log = feature
        return log

    def loss_cl(self, x1, x1_o_2, x2):
        ret_os = self.disc(x1, x1_o_2, x2) #(1565,2)
        return ret_os
