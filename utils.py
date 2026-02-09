import torch
import numpy as np
import json
from torch.utils.data import DataLoader, Subset

def k_fold_dataloaders(dataset, k=5, batch_size=32, shuffle=True, seed=42):
    n = len(dataset)
    indices = np.arange(n)

    fold_size = n // k
    folds = []

    for i in range(k):
        if i < k - 1:
            test_idx = indices[i * fold_size:(i + 1) * fold_size]
        else:
            test_idx = indices[i * fold_size:]
        
        train_idx = np.setdiff1d(indices, test_idx)

        train_loader = DataLoader(
            Subset(dataset, train_idx.tolist()),
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_RL,
            drop_last=False
        )
        test_loader = DataLoader(
            Subset(dataset, test_idx.tolist()),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_RL,
            drop_last=False
        )

        folds.append((train_loader, test_loader))
    
    return folds

def compare_evaluate_result(left, right):
    lmae, lmse, rmae, rmse = left[0], left[1], right[0], right[1]
    if lmse < rmse:
        if lmae < rmae:
            return True
        else:
            if rmse - lmse >= lmae - rmae:
                return True
            else:
                return False
    else:
        if lmae > rmae:
            return False
        else:
            if lmse - rmse >= rmae - lmae:
                return False
            else: 
                return True

def get_all_reflect_relation(data_path):
    """obtain drawing and course mapping relationships, dimensional correlation matrices
    Args:
        data_path (Str): the path where the data is stored
    Returns:
        Dict, Tensor(77,77): the dictionary stores the mapping of drawings and lessons, 
        Tensor(77, 77): dimension_adj_matrix stores 77 dimensions of interrelationships
    """    
    with open(f'{data_path}/reflect.json', 'r') as f:
        ability, dimension_to_idx = json.load(f)
    return ability, dimension_to_idx


def get_RL_data(photo_size, data_path):
    """preprocess the data and divide the training set and the test set
    Args:
        data_path (Str): the path where the data is stored
    Returns:
        _type_: the whole dataset
    """    
    with open(f'{data_path}/data.json', 'r') as f:
        tl_set = json.load(f)
    return tl_set

def collate_RL(data):
    """merges a list of samples to form a mini-batch of Tensor(s).
    Args:
        data ([Tensor(), Str, List(num_classes)]): _description_
    Returns:
        Tensor(batch_size, 3, photo_size, photo_size): picture representation
        [Str]: the name of the picture
        Tensor(batch_size, num_classes): image tags
    """    
    img = [i[0] for i in data]
    name = [i[1] for i in data]
    labels = []
    for i in data:
        one_batch = []
        for j in i[-1]:
            one_batch.append(j)
        labels.append(one_batch)
    return torch.tensor(img), name, torch.tensor(labels)

def collate_data(data):
    img = torch.tensor([i[0].tolist() for i in data])
    name = [i[1] for i in data]
    labels = torch.tensor([i[-1].tolist() for i in data])
    return img, name, labels

def get_dimension_level_info(data_path):
    with open(f'{data_path}/dimension_info.json', 'r') as f:
        first_dimension_dict, second_dimension_dict = json.load(f)
    return first_dimension_dict, second_dimension_dict
