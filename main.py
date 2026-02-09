import os
import numpy as np
import torch
import argparse
import torch.optim as optim
from utils import *
from focal_loss_example import *
from Model.CEGP import CEGP
import warnings
import os
import os.path as pth
from Log import Log
import random
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    print(torch.cuda.is_available())
    torch.cuda.set_device(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    parser = argparse.ArgumentParser()

    parser.add_argument('--batchSize', type=int, default=2, help='the input batch size')
    parser.add_argument('--hidden_dim', type=int, default=100, help='the hidden state size')
    parser.add_argument('--epochs', type=int, default=30, help='the number of epochs to train for')
    
    parser.add_argument('--lr', type=float, default=0.001, help='the learning rate')
    parser.add_argument('--num_layers', type=int, default=6, help='the layers nums')
    parser.add_argument('--num_heads', type=int, default=8, help='the attention heads nums')
    parser.add_argument('--mlp_ratio', type=float, default=1, help='the ratio of hidden layers in the middle')
    parser.add_argument('--Kernel_size1', type=int, default=2, help='the first layer convolution kernel size')
    parser.add_argument('--Kernel_size2', type=int, default=2, help='the second layer convolution kernel size')
    parser.add_argument('--Stride1', type=int, default=2, help='the second layer convolution stride size')
    parser.add_argument('--Stride2', type=int, default=2, help='the second layer convolution stride size')
    parser.add_argument('--agent_nums', type=int, default=77, help='the number of agents')
    parser.add_argument('--photo_size', type=int, default=128, help='the size of photo')
    parser.add_argument('--Linear_nums', type=int, default=3, help='the number of linear layers')
    parser.add_argument('--data_path', type=str, default='unify_data/data_new', help='the path of data')
    parser.add_argument('--embedding_dim', type=int, default=300, help='the global vector dimension')
    parser.add_argument('--FocalLoss_alpha', type=float, default=2, help='the value of the alpha parameter in FocalLoss')
    parser.add_argument('--FocalLoss_gamma', type=float, default=1, help='the value of the gamma parameter in FocalLoss')
    parser.add_argument('--log_config_path', type=str, default='LogConfig.ini', help='the path of log config file')
    parser.add_argument('--log_level', type=str, default='DEBUG', help='the log level')
    parser.add_argument('--res_save_path', type=str, default='result', help='the path of result file')
    parser.add_argument('--expand_name', type=str, default=str('debug'), help='the extension names used to distinguish different experiments')
    parser.add_argument('--SEED', type=int, default=0, help='the seed for random')

    opt = parser.parse_args()
    SEED = opt.SEED
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_default_dtype(torch.float32)
    project_dir = pth.dirname(pth.realpath(__file__))
    opt.log = Log(config_path=pth.join(project_dir, opt.log_config_path), log_dir=pth.join(project_dir, 'log', opt.expand_name), 
                  res_dir=pth.join(project_dir, f'{opt.res_save_path}', f'{opt.expand_name}.txt'), adjust_dir=pth.join(project_dir, f'parameter_{opt.expand_name}.txt'), global_level=opt.log_level)
    for param, value in vars(opt).items(): opt.log.log_args(f'{param} {value}')
    print(f'opt.log.res_dir is {opt.log.res_dir}')

    tl_set = get_RL_data(opt.photo_size, opt.data_path)
    folds = k_fold_dataloaders(tl_set, k=5, batch_size=opt.batchSize, shuffle=False, seed=opt.SEED)


    embedding_dim = int((opt.photo_size - opt.Kernel_size1) // opt.Stride1 + 1)
    embedding_dim = int((embedding_dim - 2) // 2 + 1)
    embedding_dim = int((embedding_dim - opt.Kernel_size2 + 2) // opt.Stride2 + 1)
    embedding_dim = int((embedding_dim - 2) // 2 + 1)
    embedding_dim = int(embedding_dim * embedding_dim * 3)

    final_result = []
    for fold, (train_loader, test_loader) in enumerate(folds):
        model = CEGP(opt, embedding_dim).cuda()
        test_result = [1e9 for _ in range(3)]
        opt.log.log_result(f'=== Fold {fold+1} ===')
        for epoch in range(opt.epochs):
            model.train(train_loader)
            result = model.test(test_loader)
            if compare_evaluate_result(result, test_result): test_result = result
        final_result.append(test_result)
        opt.log.log_result(f'Fold {fold+1} best result is {test_result}')
    final_result = torch.mean(torch.tensor(final_result), dim=0).tolist()
    opt.log.log_result(f'final_result {final_result[0]} {final_result[1]} {final_result[2]} {final_result[3]} {final_result[4]} {final_result[5]}')