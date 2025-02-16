import warnings

from tqdm import tqdm
from loss_function import GHMCLoss
from model import *

warnings.filterwarnings("ignore")
seed = 47
set_random_seed(seed)

args = {}

hidden_size = 512
out_size = 64
dropout = 0.8
lr = 0.001

weight_decay = 1e-6
epochs = 300

reg_loss_co = 0.001
fold = 0

T = 2.0
num_tasks = 4

L_t = torch.zeros(num_tasks)
L_t_minus_1 = torch.ones(num_tasks)

graph = []

def main(tr, te, lncRNA_miRNA_matrix_numpy,protein_miRNA_matrix_numpy,threshold,negative_ratio):
    all_metrics = {
        'acc': [],
        'pre': [],
        'roc': [],
        'aupr': [],
        'f1': [],
        'recall': []
    }

    global graph


    for i in range(len(tr)):
        train_index = tr[i]
        test_index = te[i]


        graph, all_meta_paths = create_heterograph(data_dir, lncRNA_protein_matrix, lncRNA_miRNA_matrix_numpy,
                                                   protein_miRNA_matrix_numpy)

        l_edge_list,p_edge_list = build_hypergraph(lncRNA_miRNA_matrix_numpy,protein_miRNA_matrix_numpy)

        lncRNA_miRNA_matrix = torch.from_numpy(lncRNA_miRNA_matrix_numpy).float().to(args['device'])
        protein_miRNA_matrix = torch.from_numpy(protein_miRNA_matrix_numpy).float().to(args['device'])

        l_hpGraph = Hypergraph(lncRNA_features.shape[0], l_edge_list).to(args['device'])
        p_hpGraph = Hypergraph(protein_features.shape[0], p_edge_list).to(args['device'])
        model = ICMFLPI(
            all_meta_paths=all_meta_paths,
            in_size=[lncRNA_features.shape[1], protein_features.shape[1], miRNA_features.shape[1]],
            hidden_size=hidden_size,
            out_size=out_size,
            dropout=dropout,
            mask_rna=mask_rna,
            mask_pro=mask_pro,
            mask_mi=mask_mi,
            rna_mi=lncRNA_miRNA_matrix,
            pro_mi=protein_miRNA_matrix,
            l_hpGraph=l_hpGraph,
            p_hpGraph=p_hpGraph,
        ).to(args['device'])
        # model.load_state_dict(torch.load(f"{dir}/net{i}.pth"))
        optim = torch.optim.AdamW(lr=lr, weight_decay=weight_decay, params=model.parameters())

        best_auroc = 0.0
        best_aupr = 0.0
        best_acc = 0.0
        best_f1 = 0.0
        best_precision = 0.0
        best_recall = 0.0

        mc = {
            'auroc': [],
            'aupr': [],
            'acc': [],
            'f1': [],
            "precision": [],
            'recall': [],
            'fpr':[],
            'tpr':[]
        }

        for epoch in tqdm(range(epochs)):
            out = train(model, optim, train_index)


            out = test(model, test_index)

            auroc, aupr, acc, f1, pre, recall,tpr,fpr = metric(out, label[test_index])

            mc['auroc'].append(auroc)
            mc['aupr'].append(aupr)
            mc['acc'].append(acc)
            mc['f1'].append(f1)
            mc['precision'].append(pre)
            mc['recall'].append(recall)

            if epoch / epochs > 0.9:
                if auroc > best_auroc:
                    best_auroc = auroc
                if aupr > best_aupr:
                    best_aupr = aupr
                if acc > best_acc:
                    best_acc = acc
                if f1 > best_f1:
                    best_f1 = f1
                if pre > best_precision:
                    best_precision = pre
                if recall > best_recall:
                    best_recall = recall

            # print('AUROC= %.4f | AUPR= %.4f | ACC= %.4f | F1_score= %.4f | precision= %.4f | recall= %.4f' % (
            #     auroc, aupr, acc, f1, pre, recall))
            #


        # auroc, aupr, acc, f1, pre, recall,tpr,fpr = metric(out, label[test_index])
        all_metrics['acc'].append(best_acc)
        all_metrics['pre'].append(best_precision)
        all_metrics['roc'].append(best_auroc)
        all_metrics['aupr'].append(best_aupr)
        all_metrics['f1'].append(best_f1)
        all_metrics['recall'].append(best_recall)

        print('AUROC= %.4f | AUPR= %.4f | ACC= %.4f | F1_score= %.4f | precision= %.4f | recall= %.4f' % (
            best_auroc, best_aupr, best_acc, best_f1, best_precision, best_recall))


    roc = np.mean(all_metrics['roc'])
    aupr = np.mean(all_metrics['aupr'])
    acc = np.mean(all_metrics['acc'])
    f1 = np.mean(all_metrics['f1'])
    pre = np.mean(all_metrics['pre'])
    recall = np.mean(all_metrics['recall'])

    print('AUROC= %.4f | AUPR= %.4f | ACC= %.4f | F1_score= %.4f | precision= %.4f | recall= %.4f' % (
        roc, aupr, acc, f1, pre, recall))

def test(model, test_index):

    global L_t, L_t_minus_1
    model.eval()
    out, rna_cross_loss, pro_cross_loss, mi_cross_loss, lncRNA_h, protein_h, miRNA_h, rna_embs, protein_embs, miRNA_embs = model(
        graph,
        node_feature,
        test_index,
        data)

    return out

def train(model, optim, train_index):
    global L_t, L_t_minus_1

    model.train()
    out, rna_cross_loss, pro_cross_loss, mi_cross_loss, lncRNA_h, protein_h, miRNA_h, rna_embs, protein_embs, miRNA_embs = model(
        graph,
        node_feature,
        train_index,
        data)

    reg = get_L2reg(model.parameters())

    task_losses = []
    criterion = GHMCLoss(bins=10, momentum=0.75)

    task_losses.append(criterion(out, label[train_index].reshape(-1).long()))
    task_losses.append(rna_cross_loss)
    task_losses.append(pro_cross_loss)
    task_losses.append(mi_cross_loss)

    L_t[0] = task_losses[0].item()
    L_t[1] = task_losses[1].item()
    L_t[2] = task_losses[2].item()
    L_t[3] = task_losses[3].item()

    w_k = L_t / L_t_minus_1
    lambda_k = len(L_t) * torch.exp(w_k / T) / torch.sum(torch.exp(w_k / T))

    loss = 0
    for i in range(len(L_t)):
        loss += lambda_k[i] * task_losses[i]

    L_t_minus_1 = L_t.clone()

    loss = loss + reg_loss_co * reg
    # print("train loss: ", loss.item() )
    optim.zero_grad()
    loss.backward()
    optim.step()


    return out


data_dir = "data1/"
args['device'] = "cuda:0" if torch.cuda.is_available() else "cpu"

nsm = [3]
thresholds = [3]
for threshold in thresholds:
    for negative_sample_multiplier in nsm:

        lncRNA_protein_dataset, lncRNA_protein_matrix, lncRNA_miRNA_matrix, protein_miRNA_matrix = load_dataset(data_dir,
                                                                                                                negative_sample_multiplier=negative_sample_multiplier,
                                                                                                                threshold=threshold)

        lncRNA_miRNA_matrix, protein_miRNA_matrix = filter_interactions(lncRNA_miRNA_matrix, protein_miRNA_matrix, threshold)
        lncRNA_miRNA_matrix = remove_half_interactions(lncRNA_miRNA_matrix,0.8)


        lncRNA_features = torch.randn(lncRNA_protein_matrix.shape[0], 180).to(args['device'])
        protein_features = torch.randn(lncRNA_protein_matrix.shape[1], 180).to(args['device'])
        miRNA_features = torch.randn(lncRNA_miRNA_matrix.shape[1], 180).to(args['device'])

        node_feature = [lncRNA_features, protein_features, miRNA_features]



        lncRNA_GK_similarity_matrix = np.loadtxt(data_dir + 'lncRNA_GK_similarity_matrix.txt')
        miRNA_GK_similarity_matrix = np.loadtxt(data_dir + 'miRNA_GK_similarity_matrix.txt')
        protein_GO_similarity_matrix = np.loadtxt(data_dir + 'proteinGO.txt')
        protein_GK_similarity_matrix = np.loadtxt(data_dir + 'protein_GK_similarity_matrix.txt')

        lnc_gk = neighborhood(lncRNA_GK_similarity_matrix, 20)
        pro_go = neighborhood(protein_GO_similarity_matrix, 10)
        pro_gk = neighborhood(protein_GK_similarity_matrix, 10)
        mi_gk = neighborhood(miRNA_GK_similarity_matrix, 30)


        mask_pro_go = torch.from_numpy(pro_go).to(args['device'])
        # maks_pro_gk = torch.from_numpy(pro_gk).to(args['device'])
        # mask_pro = torch.logical_or(mask_pro_go,maks_pro_gk)

        mask_pro = mask_pro_go
        mask_rna = torch.from_numpy(lnc_gk).to(args['device'])
        mask_mi = torch.from_numpy(mi_gk).to(args['device'])

        lncRNA_protein_label = torch.tensor(lncRNA_protein_dataset[:, 2:3]).to(args['device'])

        data = lncRNA_protein_dataset

        label = lncRNA_protein_label

        train_indeces, test_indeces = get_cross(lncRNA_protein_dataset)

        main(train_indeces, test_indeces,lncRNA_miRNA_matrix,protein_miRNA_matrix,threshold,negative_sample_multiplier)

