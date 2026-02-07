import torch
import torch.nn as nn
import torch.nn.functional as F

class CoExpert(nn.Module):
    def __init__(self, opt) -> None:
        super(CoExpert, self).__init__()
        self.opt = opt

    def cal_attention_weight(self, x, y):
        x, y = x.squeeze(0), y.squeeze(0)
        score = torch.dot(x, y) / (x.norm() * y.norm() + 1e-8)
        return score.item()

    def aggregate_parameter(self, first_seconde_dimensions, dimension_represents, params_dict):
        new_params_dict = {ke:va for ke, va in params_dict.items()}
        for one_clique_dimension in first_seconde_dimensions.values():
            for i in one_clique_dimension:
                new_params_dict[i] = {}
                for ke in params_dict[i].keys():
                    acc = torch.zeros_like(params_dict[i][ke])
                    weights = []
                    for j in one_clique_dimension:
                        weight = self.cal_attention_weight(dimension_represents[i], dimension_represents[j])
                        if isinstance(weight, torch.Tensor):
                            weight = weight.item() 
                        weights.append(weight)
                    weights = torch.tensor(weights, device=acc.device)
                    weights = F.softmax(weights, dim=0)
                    for idx, j in enumerate(one_clique_dimension):
                        acc += weights[idx] * params_dict[j][ke]
                    new_params_dict[i][ke] = acc

        return new_params_dict