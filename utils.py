import csv
import os
from collections import defaultdict

import dgl
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, roc_curve, accuracy_score, \
    average_precision_score, auc, recall_score

from scipy import sparse
import torch
import random
import numpy as np


def set_random_seed(seed=0):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def setup(args, seed):
    set_random_seed(seed)
    return args



def create_heterograph(network_path,lncRNA_protein_interaction_matrix,lncRNA_miRNA_interaction_matrix,protein_miRNA_interaction_matrix):
    lncRNA_GK_similarity_matrix = np.loadtxt(network_path + 'lncRNA_GK_similarity_matrix.txt')

    lncRNA_protein = lncRNA_protein_interaction_matrix
    protein_lncRNA = lncRNA_protein.T

    lncRNA_miRNA = lncRNA_miRNA_interaction_matrix
    miRNA_lncRNA = lncRNA_miRNA.T

    miRNA_GK_similarity_matrix = np.loadtxt(network_path + 'miRNA_GK_similarity_matrix.txt')

    protein_GO_similarity_matrix = np.loadtxt(network_path + 'proteinGO.txt')
    protein_GK_similarity_matrix = np.loadtxt(network_path + 'protein_GK_similarity_matrix.txt')

    protein_miRNA = protein_miRNA_interaction_matrix
    miRNA_protein = protein_miRNA.T

    lncRNA_gk = dgl.from_scipy(sparse.csr_matrix(neighborhood(lncRNA_GK_similarity_matrix, 20)))


    lncRNA_miRNA = dgl.bipartite_from_scipy(sparse.csr_matrix(lncRNA_miRNA), 'lncRNA', 'lncRNA_miRNA', 'miRNA')
    miRNA_lncRNA = dgl.bipartite_from_scipy(sparse.csr_matrix(miRNA_lncRNA), 'miRNA', 'miRNA_lncRNA', 'lncRNA')

    lncRNA_protein = dgl.bipartite_from_scipy(sparse.csr_matrix(lncRNA_protein), 'lncRNA', 'lncRNA_protein', 'protein')
    protein_lncRNA = dgl.bipartite_from_scipy(sparse.csr_matrix(protein_lncRNA), 'protein', 'protein_lncRNA', 'lncRNA')


    protein_go = dgl.from_scipy(sparse.csr_matrix(neighborhood(protein_GO_similarity_matrix, 10)))
    protein_gk = dgl.from_scipy(sparse.csr_matrix(neighborhood(protein_GK_similarity_matrix, 10)))

    miRNA_gk = dgl.from_scipy(sparse.csr_matrix(neighborhood(miRNA_GK_similarity_matrix, 30)))
    protein_miRNA = dgl.bipartite_from_scipy(sparse.csr_matrix(protein_miRNA), 'protein', 'protein_miRNA', 'miRNA')
    miRNA_protein = dgl.bipartite_from_scipy(sparse.csr_matrix(miRNA_protein), 'miRNA', 'miRNA_protein', 'protein')



    lncRNA_heterograph = dgl.heterograph({
        ('lncRNA', 'lncRNA_GK_similarity', 'lncRNA'): lncRNA_gk.edges(),

        ('lncRNA', 'lncRNA_miRNA', 'miRNA'): lncRNA_miRNA.edges(),
        ('miRNA', 'miRNA_lncRNA', 'lncRNA'): miRNA_lncRNA.edges(),

        ('lncRNA', 'lncRNA_protein', 'protein'): lncRNA_protein.edges(),
        ('protein', 'protein_lncRNA', 'lncRNA'): protein_lncRNA.edges(),
    }).to("cuda:0")

    protein_heterograph = dgl.heterograph({
        ('protein', 'protein_GO_similarity', 'protein'): protein_go.edges(),
        ('protein','protein_GK_similarity', 'protein'): protein_gk.edges(),
        ('protein', 'protein_miRNA', 'miRNA'): protein_miRNA.edges(),

        ('miRNA', 'miRNA_protein', 'protein'): miRNA_protein.edges(),
        ('lncRNA', 'lncRNA_protein', 'protein'): lncRNA_protein.edges(),
        ('protein', 'protein_lncRNA', 'lncRNA'): protein_lncRNA.edges()
    }).to("cuda:0")

    miRNA_heterograph = dgl.heterograph({
        ('miRNA', 'miRNA_GK_similarity', 'miRNA'): miRNA_gk.edges(),
        ('miRNA', 'miRNA_protein', 'protein'): miRNA_protein.edges(),
        ('protein', 'protein_miRNA', 'miRNA'): protein_miRNA.edges(),
        ('miRNA', 'miRNA_lncRNA', 'lncRNA'): miRNA_lncRNA.edges(),
        ('lncRNA', 'lncRNA_miRNA', 'miRNA'): lncRNA_miRNA.edges(),

    }).to("cuda:0")
    graph = [lncRNA_heterograph, protein_heterograph, miRNA_heterograph]


    all_meta_paths = [[
        ['lncRNA_GK_similarity'],
        ['lncRNA_miRNA', 'miRNA_lncRNA'],
        ['lncRNA_protein', 'protein_lncRNA']
    ],
        [
            ['protein_GO_similarity'],
            # ['protein_GK_similarity'],
            ['protein_miRNA', 'miRNA_protein'],
            ['protein_lncRNA', 'lncRNA_protein'],
        ],
        [
            ['miRNA_GK_similarity'],
            ['miRNA_lncRNA', 'lncRNA_miRNA'],
            ['miRNA_protein', 'protein_miRNA']
        ],
    ]

    return graph, all_meta_paths


def load_interaction_matrix(network_path):
    lncRNA_protein_matrix = np.loadtxt(network_path + 'lncRNA_protein_interaction_matrix.csv', delimiter=',')
    lncRNA_miRNA_matrix = np.loadtxt(network_path + 'lncRNA_miRNA_interaction_matrix.csv', delimiter=',')
    protein_miRNA_matrix = np.loadtxt(network_path + 'protein_miRNA_interaction_matrix.csv', delimiter=',')
    # lncRNA_protein_matrix = np.loadtxt(network_path + 'lncRNA_protein_matrix.txt')
    return lncRNA_protein_matrix, lncRNA_miRNA_matrix, protein_miRNA_matrix


def load_features(data_path):
    rnafeat = np.loadtxt(data_path + 'lncRNA features.txt')
    rnafeat = minmax_scale(rnafeat, axis=0)

    rnafeatorch = torch.from_numpy(rnafeat).float()
    profeat = np.loadtxt(data_path + 'protein features.txt')
    profeat = minmax_scale(profeat, axis=0)
    protfeatorch = torch.from_numpy(profeat).float()
    return rnafeatorch, protfeatorch


def load_dataset(network_path,negative_sample_multiplier, threshold):
    lncRNA_protein_matrix, lncRNA_miRNA_matrix, protein_miRNA_matrix = load_interaction_matrix(network_path)
    print("lncRNA-protein shape: "+str(lncRNA_protein_matrix.shape))
    print("lncRNA-miRNA shape: "+str(lncRNA_miRNA_matrix.shape))
    print("protein-miRNA shape: "+str(protein_miRNA_matrix.shape))
    print("lncRNA-protein pair nums: " + str(len(np.where(lncRNA_protein_matrix == 1)[0])))
    print("lncRNA-miRNA pair nums: " + str(len(np.where(lncRNA_miRNA_matrix == 1)[0])))
    print("protein-miRNA pair nums: " + str(len(np.where(protein_miRNA_matrix == 1)[0])))

    positive_samples = []
    negative_samples = []

    num_lncRNA, num_protein = lncRNA_protein_matrix.shape
    num = 0
    for i in range(num_lncRNA):
        for j in range(num_protein):
            value = 0
            if lncRNA_protein_matrix[i, j] == 1:
                value = 1
                positive_samples.append([i, j, value])
            common_miRNA_count = np.sum(np.logical_and(lncRNA_miRNA_matrix[i, :], protein_miRNA_matrix[j, :]))
            if common_miRNA_count >= threshold:
                if value == 0:
                    num += 1
                    value = 1
            if value != 1:
                negative_samples.append([i, j, value])



    num_positive_samples = len(positive_samples)
    num_negative_samples_to_sample = num_positive_samples * negative_sample_multiplier

    random_negative_samples = random.sample(negative_samples, num_negative_samples_to_sample)

    final_dataset = positive_samples + random_negative_samples
    random.shuffle(final_dataset)
    final_dataset = np.array(final_dataset)

    print("common miRNA nums: "+str(num))
    print("positive samples nums: "+str(len(positive_samples)))
    print("negative samples nums: "+str(len(random_negative_samples)))

    return final_dataset,lncRNA_protein_matrix,lncRNA_miRNA_matrix,protein_miRNA_matrix


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sparse.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def scaley(y):
    return (y - y.min()) / y.max()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def get_cross(data, split=5):
    """
    :param data: dataset and label
    :return:
    testset index and trainset index
    """
    set1 = []
    set2 = []
    skf = StratifiedKFold(n_splits=split, shuffle=True,random_state=47)
    for train_index, test_index in skf.split(data[:, :2], data[:, 2:3]):
        set1.append(train_index)
        set2.append(test_index)
    return set1, set2


def metric(y_pred, y_true):
    y_true = y_true.cpu()
    fpr, tpr, rocth = roc_curve(y_true, y_pred.exp()[:, 1:].cpu().detach().numpy())
    auroc = auc(fpr, tpr)
    precision, recall, prth = precision_recall_curve(y_true, y_pred.exp()[:, 1:].cpu().detach().numpy())
    aupr = auc(recall, precision)

    f1 = f1_score(y_true, y_pred.argmax(dim=1).cpu().detach().numpy())
    acc = accuracy_score(y_true, y_pred.argmax(dim=1).cpu().detach().numpy())
    pre = average_precision_score(y_true, y_pred.argmax(dim=1).cpu().detach().numpy())
    recall = recall_score(y_true, y_pred.argmax(dim=1).cpu().detach().numpy())
    return auroc, aupr, acc, f1, pre, recall,tpr.tolist(),fpr.tolist()


def get_L2reg(parameters):
    reg = 0
    for param in parameters:
        reg += 0.5 * (param ** 2).sum()
    return reg



def neighborhood(mat, k):
    dsort = np.argsort(mat)[:, 1:k + 1]
    C = np.zeros((mat.shape[0], mat.shape[1]))
    for i in range(mat.shape[0]):
        for j in dsort[i]:
            C[i, j] = 1.0

    return C


def mask_node_features(features: torch.Tensor, noise_level: float = 0.1) -> torch.Tensor:
    num_nodes, num_features = features.shape
    mask = np.random.binomial(1, noise_level, (num_nodes, num_features))
    mask = torch.FloatTensor(mask).to('cuda:0')
    features = features * (1 - mask)
    return features


def add_noise_to_nodes(features: torch.Tensor, noise_level: float = 0.2) -> torch.Tensor:
    noise = torch.randn_like(features) * noise_level
    noisy_features = features + noise
    return noisy_features


def remove_half_interactions(interaction_matrix,drop_percent=0.5):
    interaction_indices = np.argwhere(interaction_matrix == 1)
    num_interactions_to_remove = int(len(interaction_indices) * drop_percent)
    indices_to_remove = np.random.choice(len(interaction_indices), num_interactions_to_remove, replace=False)

    modified_interaction_matrix = interaction_matrix.copy()

    for idx in indices_to_remove:
        i, j = interaction_indices[idx]
        modified_interaction_matrix[i, j] = 0

    return modified_interaction_matrix

def filter_interactions(lncRNA_miRNA_matrix, protein_miRNA_matrix, threshold=5):
    lncRNA_miRNA = np.array(lncRNA_miRNA_matrix)
    protein_miRNA = np.array(protein_miRNA_matrix)

    common_miRNAs = np.intersect1d(np.where(lncRNA_miRNA.sum(axis=0) > 0)[0], np.where(protein_miRNA.sum(axis=0) > 0)[0])

    filtered_lncRNA_miRNA = np.zeros_like(lncRNA_miRNA)
    filtered_protein_miRNA = np.zeros_like(protein_miRNA)

    filtered_lncRNA_miRNA[:, common_miRNAs] = lncRNA_miRNA[:, common_miRNAs]
    filtered_protein_miRNA[:, common_miRNAs] = protein_miRNA[:, common_miRNAs]

    lncRNA_miRNA_counts = np.sum(filtered_lncRNA_miRNA > 0, axis=1)
    protein_miRNA_counts = np.sum(filtered_protein_miRNA > 0, axis=1)

    lncRNA_keep = lncRNA_miRNA_counts >= threshold
    protein_keep = protein_miRNA_counts >= threshold

    final_lncRNA_miRNA = np.where(np.repeat(lncRNA_keep[:, np.newaxis], lncRNA_miRNA.shape[1], axis=1), filtered_lncRNA_miRNA, 0)
    final_protein_miRNA = np.where(np.repeat(protein_keep[:, np.newaxis], protein_miRNA.shape[1], axis=1), filtered_protein_miRNA, 0)

    return final_lncRNA_miRNA, final_protein_miRNA


def edge_list_to_adjacency_matrix(edge_list, num_nodes):
    adjacency_matrix = np.zeros((num_nodes[0], num_nodes[1]))

    for src, dst,label in edge_list:
        if label == 1:
            adjacency_matrix[src, dst] = 1

    return adjacency_matrix



import numpy as np


def build_hypergraph(lncRNA_miRNA_interaction_matrix, protein_miRNA_interaction_matrix):


    num_lncRNA = lncRNA_miRNA_interaction_matrix.shape[0]
    num_protein = protein_miRNA_interaction_matrix.shape[0]
    num_miRNA = lncRNA_miRNA_interaction_matrix.shape[1]

    hypergraph1 = [[] for _ in range(num_miRNA)]
    hypergraph2 = [[] for _ in range(num_miRNA)]

    for lncRNA_index in range(num_lncRNA):
        for miRNA_index in range(num_miRNA):
            if lncRNA_miRNA_interaction_matrix[lncRNA_index, miRNA_index] > 0:
                hypergraph1[miRNA_index].append(lncRNA_index)

    for protein_index in range(num_protein):
        for miRNA_index in range(num_miRNA):
            if protein_miRNA_interaction_matrix[protein_index, miRNA_index] > 0:
                hypergraph2[miRNA_index].append(protein_index)

    hypergraph1 = [edge for edge in hypergraph1 if len(edge) >= 2]
    hypergraph2 = [edge for edge in hypergraph2 if len(edge) >= 2]

    return hypergraph1, hypergraph2


