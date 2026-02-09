import torch
import torch.nn as nn
from focal_loss_example import FocalLoss
from utils import *
import torch.optim as optim
from Model.CNN import CNN
from Model.Expert import Expert
from Model.CoExpert import CoExpert
from Model.CAGNN import CAGNN

class CEGP(nn.Module):
    def __init__(self, opt, embedding_dim) -> None:
        super(CEGP, self).__init__()
        self.opt = opt
        self.dimension_map, self.dimension_to_idx = get_all_reflect_relation(opt.data_path)
        self.first_dimension_dict, self.second_dimension_dict = get_dimension_level_info(opt.data_path)

        self.common_shared_model = CNN(embedding_dim, opt).cuda()
        self.represent_model = CAGNN(opt, self.dimension_to_idx, len(self.dimension_map)).cuda()
        
        self.dimensions_embedding = nn.Embedding(opt.dim_nums, opt.embedding_dim).cuda()
        self.dimensions_embed = self.dimensions_embedding(torch.LongTensor([_ for _ in range(opt.dim_nums)]).cuda()).cuda()
        self.collaborative_expert = CoExpert(opt)
        self.experts = {ke : Expert(opt, embedding_dim + opt.embedding_dim, len(va)) for ke, va in self.dimension_map.items()}
        self.experts_optimizer = optim.Adam((param for expert in self.experts.values() for param in expert.parameters()), lr=opt.lr)
        self.comomn_optimizer = optim.Adam(list(self.common_shared_model.parameters()) + list(self.represent_model.parameters()), lr=opt.lr)

        self.loss_func1 = FocalLoss(opt.FocalLoss_alpha, opt.FocalLoss_gamma)
        self.loss_func2 = nn.MSELoss()
        self.loss_func3 = nn.L1Loss()

    def evaluate(self, score, label):
        score, label = torch.tensor(score), torch.tensor(label)
        mse = torch.mean(torch.abs(score - label) ** 2)
        mae = torch.mean(torch.abs(score - label))
        rmse = np.sqrt(mse)
        return mae, mse, rmse

    def train(self, data):
        for img, _, labels in data:
            common_dimension_ability = self.common_shared_model(img.cuda())
            all_dimension_represent = self.represent_model(self.dimensions_embed)
            self.opt.log.log_info(f'all_dimension_represent.shape is {all_dimension_represent.shape}')
            dimension_represents = {}
            total_loss = 0
            for ke, one_expert in self.experts.items():
                expert_to_dimension_idx = torch.tensor([self.dimension_to_idx[_] for _ in self.dimension_map[ke]]).cuda()
                CAGNN_ouput = torch.mean(torch.index_select(input=all_dimension_represent, dim=0, index=expert_to_dimension_idx), dim=0, keepdim=True).cuda()
                dimension_represents[ke] = CAGNN_ouput
                CAGNN_ouput = CAGNN_ouput.repeat(common_dimension_ability.size(0), 1).cuda()
                expert_sc = torch.cat((common_dimension_ability, CAGNN_ouput), dim=1).cuda()
                expert_out = one_expert(expert_sc)
                expert_labels = torch.index_select(input=labels.cuda(), dim=1, index=expert_to_dimension_idx).cuda()
                loss1, loss2, loss3 = [_(expert_out, expert_labels) for _ in (self.loss_func1, self.loss_func2, self.loss_func3)]
                loss = loss1 + loss2 + loss3
                total_loss += loss
            self.experts_optimizer.zero_grad()
            self.opt.log.log_info(f'loss is {loss}')
            total_loss.backward(retain_graph=True)
            self.experts_optimizer.step()

            t2_loss = 0
            for ke, one_expert in self.experts.items():
                expert_to_dimension_idx = torch.tensor([self.dimension_to_idx[_] for _ in self.dimension_map[ke]]).cuda()
                CAGNN_ouput = torch.mean(torch.index_select(input=all_dimension_represent, dim=0, index=expert_to_dimension_idx), dim=0, keepdim=True).cuda()
                dimension_represents[ke] = CAGNN_ouput
                CAGNN_ouput = CAGNN_ouput.repeat(common_dimension_ability.size(0), 1).cuda()
                expert_sc = torch.cat((common_dimension_ability, CAGNN_ouput), dim=1).cuda()
                expert_out = one_expert(expert_sc)
                expert_labels = torch.index_select(input=labels.cuda(), dim=1, index=expert_to_dimension_idx).cuda()
                loss1, loss2, loss3 = [_(expert_out, expert_labels) for _ in (self.loss_func1, self.loss_func2, self.loss_func3)]
                loss = loss1 + loss2 + loss3
                t2_loss += loss
            self.comomn_optimizer.zero_grad()
            self.opt.log.log_info(f'loss is {loss}')
            t2_loss.backward(retain_graph=True)
            self.comomn_optimizer.step()

            model_params_dict = {ke:{name: param for name, param in va.state_dict().items() if "fc3" not in name} for ke, va in self.experts.items()}
            self.opt.log.log_weight(f'old param is {model_params_dict}')
            new_params_dict = self.collaborative_expert.aggregate_parameter(self.first_dimension_dict, dimension_represents, model_params_dict)
            self.opt.log.log_weight(f'new param is {new_params_dict}')
            for clinet_name, one_expert in self.experts.items(): one_expert.load_state_dict(new_params_dict[clinet_name],strict=False)

    def test(self, data):
        expert_true_list, expert_rec_list = {}, {}
        true_list, rec_list = [[0] * self.opt.dim_nums for _ in range(len(data) * self.opt.batchSize)], [[0] * self.opt.dim_nums for _ in range(len(data) * self.opt.batchSize)]
        for ke, one_expert in self.experts.items():
            expert_true_list[ke], expert_rec_list[ke] = [], []
            expert_to_dimension_idx = torch.tensor([self.dimension_to_idx[_] for _ in self.dimension_map[ke]]).cuda()
            all_dimension_represent = self.represent_model(self.dimensions_embed)
            CAGNN_ouput = torch.mean(torch.index_select(input=all_dimension_represent, dim=0, index=expert_to_dimension_idx), dim=0, keepdim=True).cuda()
            CAGNN_ouput = CAGNN_ouput.repeat(self.opt.batchSize, 1).cuda()
            for idx, (img, name, labels) in enumerate(data):
                cur_batch_size = img.size(0)
                common_dimension_ability = self.common_shared_model(img.cuda())
                if CAGNN_ouput.size(0) != cur_batch_size: CAGNN_ouput = CAGNN_ouput[:cur_batch_size]
                expert_sc = torch.cat((common_dimension_ability, CAGNN_ouput), dim=1).cuda()
                expert_out = one_expert(expert_sc)
                expert_labels = torch.index_select(input=labels.cuda(), dim=1, index=expert_to_dimension_idx).cuda()
                expert_true_list[ke] += expert_labels.tolist()
                expert_rec_list[ke] += expert_out.tolist()

                lidx, ridx = min([self.dimension_to_idx[_] for _ in self.dimension_map[ke]]), max([self.dimension_to_idx[_] for _ in self.dimension_map[ke]])
                texpert_labels, texpert_out = expert_labels.tolist(), expert_out.tolist()
                for cid in range(cur_batch_size):
                    for tid in range(lidx, ridx + 1, 1):
                        true_list[idx * self.opt.batchSize + cid][tid] = texpert_labels[cid][tid - lidx]
                        rec_list[idx * self.opt.batchSize + cid][tid] = texpert_out[cid][tid - lidx]

            mae, mse, rmse = self.evaluate(expert_rec_list[ke], expert_true_list[ke])
            self.opt.log.log_result(f'{ke} {mae} {mse} {rmse}')
        mae, mse, rmse = self.evaluate(rec_list, true_list)
        self.opt.log.log_result(f'total {mae} {mse} {rmse}')
        return [mae, mse, rmse]