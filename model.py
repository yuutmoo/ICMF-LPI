# -*- coding: utf-8 -*-

from dgl.nn.pytorch import GraphConv


from Layer import *
from utils import *

device = "cuda:0" if torch.cuda.is_available() else "cpu"

EPS = 1e-15

import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=32):
        super(SemanticAttention, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size).apply(init),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False).apply(init)
        )

    def forward(self, z):
        w = self.project(z).mean(0)
        beta = torch.softmax(w, dim=0)
        beta = beta.expand((z.shape[0],) + beta.shape)
        return (beta * z).sum(1)


class HANLayer(nn.Module):

    def __init__(self, meta_paths, in_size, out_size, layer_num_heads, dropout):

        super(HANLayer, self).__init__()
        self.gat_layer1 = GraphConv(in_size, out_size, activation=F.relu, allow_zero_in_degree=True).apply(init)
        # self.gat_layer2 = GraphConv(out_size, out_size, activation=F.relu, allow_zero_in_degree=True).apply(init)
        self.semantic_attention = SemanticAttention(in_size=out_size * layer_num_heads)
        self.meta_paths = list(meta_path for meta_path in meta_paths)
        self._cached_graph = None
        self.dropout = dropout
        self._cached_coalesced_graph = {}

    def forward(self, g, h):
        semantic_embeddings = []
        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()
            for meta_path in self.meta_paths:
                self._cached_coalesced_graph[tuple(meta_path)] = dgl.metapath_reachable_graph(
                    g, meta_path)
        for i, meta_path in enumerate(self.meta_paths):
            new_g = self._cached_coalesced_graph[tuple(meta_path)]
            embedding = self.gat_layer1(new_g, h)
            embedding = F.dropout(embedding, inplace=False, p=self.dropout)
            semantic_embeddings.append(embedding.flatten(1))

        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)

        return self.semantic_attention(semantic_embeddings), semantic_embeddings  


class HAN(nn.Module):
    def __init__(self, meta_paths, in_size, hidden_size, out_size, dropout, num_heads=1):
        super(HAN, self).__init__()
        self.layers = nn.ModuleList()
        self.predict = nn.Linear(hidden_size * num_heads, out_size, bias=False).apply(init)
        self.han = HANLayer(meta_paths, in_size, hidden_size, num_heads, dropout)

    def forward(self, g, h):
        h, se_embeddings = self.han(g, h)
        return self.predict(h), se_embeddings


class MLP(nn.Module):
    def __init__(self, nfeat):
        super(MLP, self).__init__()
        self.MLP = nn.Sequential(
            nn.Linear(nfeat, 32, bias=False).apply(init),
            nn.ELU(),
            nn.Linear(32, 2, bias=False),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        output = self.MLP(x)
        return output


def _sim(z1: torch.Tensor, z2: torch.Tensor):
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return torch.mm(z1, z2.t())




class MultiHeadAttentionAggregator(nn.Module):
    def __init__(self, lncRNA_dim, miRNA_dim, protein_dim, hidden_dim, num_heads):
        super(MultiHeadAttentionAggregator, self).__init__()

        assert lncRNA_dim == miRNA_dim == protein_dim, "lncRNA_dim, miRNA_dim, protein_dim must be equal"

        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads

        self.lncRNA_linear = nn.Linear(lncRNA_dim, hidden_dim)
        self.miRNA_linear = nn.Linear(miRNA_dim, hidden_dim)
        self.protein_linear = nn.Linear(protein_dim, hidden_dim)

        self.q_linear_l = nn.Linear(hidden_dim, hidden_dim)
        self.k_linear_l = nn.Linear(hidden_dim, hidden_dim)
        self.v_linear_l = nn.Linear(hidden_dim, hidden_dim)

        self.q_linear_p = nn.Linear(hidden_dim, hidden_dim)
        self.k_linear_p = nn.Linear(hidden_dim, hidden_dim)
        self.v_linear_p = nn.Linear(hidden_dim, hidden_dim)

        self.ln1 = nn.LayerNorm(lncRNA_dim // 2)
        self.ln2 = nn.LayerNorm(protein_dim // 2)

        self.mlp_lncRNA = nn.Sequential(
            nn.Linear(hidden_dim * 2, lncRNA_dim // 2),
        )

        self.mlp_protein = nn.Sequential(
            nn.Linear(hidden_dim * 2, protein_dim // 2),
        )
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, lncRNA_embedding, miRNA_embedding, protein_embedding, lncRNA_miRNA_matrix, protein_miRNA_matrix):
        num_lncRNA = lncRNA_embedding.size(0)
        num_miRNA = miRNA_embedding.size(0)
        num_protein = protein_embedding.size(0)

        lncRNA_transformed = self.lncRNA_linear(lncRNA_embedding)
        miRNA_transformed = self.miRNA_linear(miRNA_embedding)
        protein_transformed = self.protein_linear(protein_embedding)

        Q_lncRNA = self.q_linear_l(lncRNA_transformed).view(num_lncRNA, self.num_heads, self.head_dim).transpose(0, 1)
        K_lncRNA = self.k_linear_l(miRNA_transformed).view(num_miRNA, self.num_heads, self.head_dim).transpose(0, 1)
        V_lncRNA = self.v_linear_l(miRNA_transformed).view(num_miRNA, self.num_heads, self.head_dim).transpose(0, 1)

        Q_protein = self.q_linear_p(protein_transformed).view(num_protein, self.num_heads, self.head_dim).transpose(0,
                                                                                                                    1)
        K_protein = self.k_linear_p(miRNA_transformed).view(num_miRNA, self.num_heads, self.head_dim).transpose(0, 1)
        V_protein = self.v_linear_p(miRNA_transformed).view(num_miRNA, self.num_heads, self.head_dim).transpose(0, 1)

        attention_scores_lncRNA = torch.matmul(Q_lncRNA, K_lncRNA.transpose(-2, -1)) / self.head_dim ** 0.5
        attention_scores_lncRNA = attention_scores_lncRNA.masked_fill(
            lncRNA_miRNA_matrix.unsqueeze(0).expand(self.num_heads, -1, -1) == 0, -1e9)
        attention_weights_lncRNA = F.softmax(attention_scores_lncRNA, dim=-1)

        attention_scores_protein = torch.matmul(Q_protein, K_protein.transpose(-2, -1)) / self.head_dim ** 0.5
        attention_scores_protein = attention_scores_protein.masked_fill(
            protein_miRNA_matrix.unsqueeze(0).expand(self.num_heads, -1, -1) == 0, -1e9)
        attention_weights_protein = F.softmax(attention_scores_protein, dim=-1)


        context_lncRNA = torch.matmul(attention_weights_lncRNA, V_lncRNA).transpose(0, 1).contiguous().view(num_lncRNA,
                                                                                                            self.hidden_dim)
        context_lncRNA = torch.relu(context_lncRNA)
        context_protein = torch.matmul(attention_weights_protein, V_protein).transpose(0, 1).contiguous().view(
            num_protein, self.hidden_dim)
        context_protein = torch.relu(context_protein)

        attention_weights_lncRNA = attention_weights_lncRNA.permute(1, 0, 2)
        attention_weights_protein = attention_weights_protein.permute(1, 0, 2)

        mutual_weights = torch.einsum('bhm,phm->bp', attention_weights_lncRNA,
                                      attention_weights_protein) / self.num_heads

        final_context_lncRNA = context_lncRNA + torch.matmul(mutual_weights, context_protein)
        final_context_protein = context_protein + torch.matmul(mutual_weights.transpose(0, 1), context_lncRNA)

        final_context_lncRNA = torch.cat((lncRNA_transformed, final_context_lncRNA), dim=-1)
        final_context_protein = torch.cat((final_context_protein, protein_transformed), dim=-1)


        # MLP for final output
        output_embedding_lncRNA = self.mlp_lncRNA(final_context_lncRNA)
        output_embedding_protein = self.mlp_protein(final_context_protein)
        output_embedding_lncRNA = self.ln1(output_embedding_lncRNA)
        output_embedding_protein = self.ln2(output_embedding_protein)
        return output_embedding_lncRNA, output_embedding_protein


class ICMFLPI(nn.Module):
    def __init__(self, all_meta_paths, in_size, hidden_size, out_size, dropout, mask_rna, mask_pro, mask_mi, rna_mi,
                 pro_mi,l_hpGraph,p_hpGraph):
        super(ICMFLPI, self).__init__()
        self.mask_rna = mask_rna
        self.mask_pro = mask_pro
        self.rna_mi = rna_mi
        self.pro_mi = pro_mi
        self.mask_mi = mask_mi

        self.l_hpGraph = l_hpGraph
        self.p_hpGraph = p_hpGraph
        self.l_hpNet = HGNNPConv(out_size, 32,drop_rate=0.1).to("cuda:0")
        self.p_hpNet = HGNNPConv(out_size, 32,drop_rate=0.1).to("cuda:0")


        self.L_HAN = HAN(all_meta_paths[0], in_size[0], hidden_size, out_size, dropout)
        self.P_HAN = HAN(all_meta_paths[1], in_size[1], hidden_size, out_size, dropout)
        self.M_HAN = HAN(all_meta_paths[2], in_size[2], hidden_size, out_size, dropout)

        self.local_projector = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.PReLU(),
                                             nn.Linear(hidden_size, hidden_size))
        self.rna_attention = nn.Linear(hidden_size, 1)
        self.protein_attention = nn.Linear(hidden_size, 1)
        self.miRNA_attention = nn.Linear(hidden_size, 1)


        self.MLP = MLP(out_size)
        self.attentionAggregator = MultiHeadAttentionAggregator(out_size, out_size, out_size, 32, 2)
        self.weight = nn.Parameter(torch.Tensor(hidden_size, hidden_size), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):

        for model in self.local_projector:
            if isinstance(model, nn.Linear):
                model.apply(init)

    def local_loss(self, z1: torch.Tensor, z2: torch.Tensor, mask):
        h1 = self.local_projector(z1)
        h2 = self.local_projector(z2)

        loss = self.infonce(h1, h2, mask)

        return loss

    def infonce(self, z1, z2, mask, tau=0.2):
        f = lambda x: torch.exp(x / tau)
        sim_intra = f(_sim(z1, z1))
        sim_inter = f(_sim(z1, z2))

        loss = -torch.log(
            (sim_inter * mask).sum(1) /
            (sim_intra.sum(1) + sim_inter.sum(1) - (sim_intra * mask).sum(1))
        )
        return loss.mean()


    def metapath_contrast(self, rna_embs, protein_embs, miRNA_embs):
        rna_contrastive_losses = []
        pro_contrastive_losses = []
        mi_contrastive_losses = []

        for i in range(rna_embs.shape[1]):
            for j in range(i, rna_embs.shape[1]):
                loss = self.contrast(rna_embs[:, i], rna_embs[:, j], self.mask_rna)
                rna_contrastive_losses.append(loss)

        for i in range(protein_embs.shape[1]):
            for j in range(i, protein_embs.shape[1]):
                loss = self.contrast(protein_embs[:, i], protein_embs[:, j], self.mask_pro)
                pro_contrastive_losses.append(loss)

        for i in range(miRNA_embs.shape[1]):
            for j in range(i, miRNA_embs.shape[1]):
                loss = self.contrast(miRNA_embs[:, i], miRNA_embs[:, j], self.mask_mi)
                mi_contrastive_losses.append(loss)


        rna_contrastive_losses = sum(rna_contrastive_losses) / len(rna_contrastive_losses)
        pro_contrastive_losses = sum(pro_contrastive_losses) / len(pro_contrastive_losses)
        mi_contrastive_losses = sum(mi_contrastive_losses) / len(mi_contrastive_losses)
        return rna_contrastive_losses, pro_contrastive_losses, mi_contrastive_losses

    def contrast(self, x1, x2, mask):
        z1 = add_noise_to_nodes(x1)
        z2 = add_noise_to_nodes(x2)

        local_loss = (self.metapath_loss(z1, z2, mask) + self.metapath_loss(z2, z1, mask)) / 2

        return local_loss

    def metapath_loss(self, z1: torch.Tensor, z2: torch.Tensor, mask):
        h1 = self.local_projector(z1)
        h2 = self.local_projector(z2)
        loss = self.infonce(h1, h2, mask)
        return loss

    def forward(self, graph, h, dateset_index, data):

        lncRNA_h, rna_embs = self.L_HAN(graph[0], h[0])
        protein_h, protein_embs = self.P_HAN(graph[1], h[1])
        miRNA_h, miRNA_embs = self.M_HAN(graph[2], h[2])

        rna_contrastive_losses, pro_contrastive_losses, mi_contrastive_losses = self.metapath_contrast(rna_embs,
                                                                                                       protein_embs,
                                                                                                       miRNA_embs)


        lncRNA_mi_h, protein_mi_h = self.attentionAggregator(lncRNA_h, miRNA_h, protein_h, self.rna_mi, self.pro_mi)

        lncRNA_hyper = self.l_hpNet(lncRNA_h,self.l_hpGraph)
        protein_hyper = self.p_hpNet(protein_h,self.p_hpGraph)
        a = 0.6
        lncRNA_mi_h = lncRNA_mi_h*a + lncRNA_hyper * (1-a)
        protein_mi_h = protein_mi_h*a + protein_hyper * (1-a)

        feature = torch.cat((lncRNA_mi_h[data[:, :1]], protein_mi_h[data[:, 1:2]]), dim=2).squeeze(1)
        # feature = torch.cat((lncRNA_h[data[:, :1]], protein_h[data[:, 1:2]]), dim=2).squeeze(1)
        pred1 = self.MLP(feature[dateset_index])


        return pred1, rna_contrastive_losses, pro_contrastive_losses, mi_contrastive_losses, lncRNA_h, protein_h, miRNA_h, rna_embs, protein_embs, miRNA_embs



def init(i):
    if isinstance(i, nn.Linear):
        torch.nn.init.xavier_uniform_(i.weight)


