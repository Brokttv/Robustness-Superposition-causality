import torch
 
 
def L2_attack(input, model, label, criterion, epsilon_factor=0.1):
    input = input.clone()
    noise = torch.randn_like(input) * 1e-4
    input = input + noise
    input.requires_grad_(True)
 
    output = model(input)
    loss_n = criterion(output, label)
    loss_n.backward()
 
    norm = torch.linalg.norm(input, ord=2, dim=1)
    avrg_norm = norm.mean()
    epsilon = avrg_norm * epsilon_factor
 
    grad = input.grad
    grad_norm = torch.linalg.norm(grad, ord=2, dim=1, keepdim=True).clamp(min=1e-8)
    unit_direction = grad / grad_norm
 
    perturbation = epsilon * unit_direction
    adversarial = input + perturbation
 
    return adversarial.detach()
 
 
def adv_train(dataloader, criterion, optimizer, model, device):
    model.train()
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        attack = L2_attack(x, model, y, criterion, epsilon_factor=0.1)
        preds_adv = model(attack)
        preds_clean = model(x)
        loss_adv = criterion(preds_adv, y)
        loss_clean = criterion(preds_clean, y)
        loss_fn = 0.5 * loss_adv + 0.5 * loss_clean
 
        optimizer.zero_grad()
        loss_fn.backward()
        optimizer.step()
 
 
def clean_train(dataloader, criterion, optimizer, model, device):
    model.train()
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        preds = model(x)
        loss = criterion(preds, y)
 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
 
 
def test(dataloader, criterion, model, device):
    model.eval()
    loss_attack = 0.0
    loss_clean = 0.0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        attack = L2_attack(x, model, y, criterion, epsilon_factor=0.1)
 
        with torch.inference_mode():
            preds_adv = model(attack)
            loss_attack += criterion(preds_adv, y).item()
            preds_clean = model(x)
            loss_clean += criterion(preds_clean, y).item()
 
    avg_loss_adv = loss_attack / len(dataloader)
    avg_loss_clean = loss_clean / len(dataloader)
    local_vuln = avg_loss_adv / avg_loss_clean
 
    return local_vuln
