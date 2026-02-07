import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import get_dimension_level_info

class SimpleGCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)
    
    def forward(self, x, adj):
        if adj.sum() == 0:
            return self.linear(x)
        deg = adj.sum(dim=1, keepdim=True) + 1e-6
        norm_adj = adj / deg
        return F.relu(self.linear(norm_adj @ x))


class CAGNN(nn.Module):
    def __init__(self, opt, dimension_to_idx, cliques_num) -> None:
        super(CAGNN, self).__init__()
        self.opt = opt
        self.dimension_nums = opt.agent_nums
        self.dimension_to_idx = dimension_to_idx
        self.intra_gcn = SimpleGCNLayer(opt.embedding_dim, opt.embedding_dim)
        self.inter_gcn = SimpleGCNLayer(opt.embedding_dim, opt.embedding_dim)
        self.fusion = nn.Linear(opt.embedding_dim * 2, opt.embedding_dim)
        self.cliques_num = cliques_num
        self.adj_construct()

    def adj_construct(self):
        self.intra_adj_list = []
        first_dimension_dict, second_dimension_dict = get_dimension_level_info(self.opt.data_path)
        self.clique_assignments = torch.zeros(self.dimension_nums, dtype=torch.long).cuda()
        for idx, va in enumerate(second_dimension_dict.values()):
            indices = [self.dimension_to_idx[name] for name in va]
            self.clique_assignments[indices] = idx
        self.clique_graph_adj = torch.zeros((len(second_dimension_dict), len(second_dimension_dict))).float().cuda()
        second_dimension_to_idx = {ke:idx for idx, ke in enumerate(second_dimension_dict.keys())}
        for va in first_dimension_dict.values():
            indices = torch.tensor([second_dimension_to_idx[i] for i in va], device=self.clique_graph_adj.device)
            self.clique_graph_adj[indices.unsqueeze(1), indices] = 1.0
        self.clique_graph_adj.fill_diagonal_(0)

        for i in range(self.cliques_num):
            idx = (self.clique_assignments == i).nonzero(as_tuple=True)[0]
            sz = idx.size(0)
            A = torch.ones(sz, sz) - torch.eye(sz) if sz > 1 else torch.zeros(sz, sz)
            self.intra_adj_list.append((idx, A.cuda()))


    def forward(self, node_feats):
        h_intra_all = torch.zeros_like(node_feats)
        clique_reps = []

        for i in range(self.cliques_num):
            idx, adj = self.intra_adj_list[i]
            if len(idx) == 0:
                clique_reps.append(torch.zeros(node_feats.shape[1]).cuda())
                continue
            h_c = node_feats[idx]
            h_out = self.intra_gcn(h_c, adj)
            self.opt.log.log_info(f'CAGNN h_out is {h_out}')
            h_intra_all[idx] = h_out
            clique_reps.append(h_out.mean(dim=0).cuda())

        clique_reps = torch.stack(clique_reps)  # [num_cliques, hidden_dim]
        clique_reps_out = self.inter_gcn(clique_reps, self.clique_graph_adj)
        self.opt.log.log_info(f'CAGNN clique_reps_out is {clique_reps_out}')

        h_inter_all = torch.zeros_like(h_intra_all).cuda()
        for i in range(node_feats.shape[0]):
            clique_id = self.clique_assignments[i]
            h_inter_all[i] = clique_reps_out[clique_id]

        h_concat = torch.cat([h_intra_all, h_inter_all], dim=1)
        h_final = self.fusion(h_concat)
        self.opt.log.log_info(f'CAGNN h_final is {h_final}')
        return h_final

