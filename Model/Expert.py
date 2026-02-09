import torch
import torch.nn as nn

class Expert(nn.Module):
    def __init__(self, opt, input_dim, output_dim) -> None:
        super(Expert, self).__init__()
        self.opt = opt
        self.fc1 = nn.Linear(input_dim, opt.hidden_dim).cuda()
        self.fc2 = nn.Linear(opt.hidden_dim, 50).cuda()
        self.fc3 = nn.Linear(50, output_dim).cuda()

    def forward(self, input):
        out = self.fc1(input)
        out = self.fc2(out)
        out = self.fc3(out)
        out = torch.sigmoid(out)
        return out