import torch
 
 
def create_data(sparsity: float, num_samples: int):
    values = torch.rand(num_samples, 100)
    mask = values > sparsity
    input_data = values * mask.float()
    target = input_data.clone()
    return input_data, target
 
