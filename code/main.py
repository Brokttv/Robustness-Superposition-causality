import argparse
import gc
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from model import Model
from data import create_data
from train import clean_train, adv_train, test
from utils import set_seed, superposition


def get_dataloaders(sparsity, num_train, num_test, batch_size):
    train_input, train_target = create_data(sparsity=sparsity, num_samples=num_train)
    test_input, test_target = create_data(sparsity=sparsity, num_samples=num_test)
    train_loader = DataLoader(TensorDataset(train_input, train_target),
                              batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_input, test_target),
                             batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def run_experiment_32(sparsity_levels, epochs, batch_size, criterion, device):
    """3.2: Superposition --> Robustness
    Train all models cleanly at varying sparsity, test on attacks.
    Vulnerability = local_vuln(S) / local_vuln(S=0)
    """
    results = {}
    baseline = None

    for sparsity in sparsity_levels:
        train_loader, test_loader = get_dataloaders(sparsity, 5000, 2000, batch_size)

        model = Model(n=20, m=100).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        for epoch in range(epochs + 1):
            if epoch % 1000 == 0:
                print(f"  [Sparsity {sparsity}] Epoch {epoch}/{epochs}")
            clean_train(train_loader, criterion, optimizer, model, device)

        local_vuln = test(test_loader, criterion, model, device)
        density = superposition(model)

        if sparsity == 0.0:
            baseline = local_vuln

        results[sparsity] = {
            "vulnerability": local_vuln / baseline,
            "density": density
        }
        print(f"  Sparsity {sparsity} done | vulnerability: {local_vuln / baseline:.4f} | density: {density:.4f}")

        del model, optimizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()

    return results


def run_experiment_33(sparsity_levels, epochs, batch_size, criterion, device):
    """3.3: Robustness --> Superposition
    For each sparsity, train one clean and one adversarial model.
    Metric = density_clean(S) - density_adv(S)
    """
    results = {}

    for sparsity in sparsity_levels:
        train_loader, _ = get_dataloaders(sparsity, 5000, 2000, batch_size)

        # --- Clean model ---
        clean_model = Model(n=20, m=100).to(device)
        clean_optimizer = torch.optim.AdamW(clean_model.parameters(), lr=1e-3)
        for epoch in range(epochs + 1):
            if epoch % 1000 == 0:
                print(f"  [Sparsity {sparsity}] Clean epoch {epoch}/{epochs}")
            clean_train(train_loader, criterion, clean_optimizer, clean_model, device)
        density_clean = superposition(clean_model)

        # --- Adversarial model ---
        adv_model = Model(n=20, m=100).to(device)
        adv_optimizer = torch.optim.AdamW(adv_model.parameters(), lr=1e-3)
        for epoch in range(epochs + 1):
            if epoch % 1000 == 0:
                print(f"  [Sparsity {sparsity}] Adv epoch {epoch}/{epochs}")
            adv_train(train_loader, criterion, adv_optimizer, adv_model, device)
        density_adv = superposition(adv_model)

        density_diff = density_clean - density_adv

        results[sparsity] = {
            "density_clean": density_clean,
            "density_adv": density_adv,
            "density_diff": density_diff
        }
        print(f"  Sparsity {sparsity} done | density_clean: {density_clean:.4f} | density_adv: {density_adv:.4f} | diff: {density_diff:.4f}")

        del clean_model, clean_optimizer, adv_model, adv_optimizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()

    return results


def main():
    parser = argparse.ArgumentParser(description="Superposition-Robustness Experiments")
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        choices=["superposition->robustness", "robustness->superposition"],
        help="Which experiment to run: 'superposition->robustness' (3.2) or 'robustness->superposition' (3.3)"
    )
    parser.add_argument("--epochs", type=int, default=1900)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=52)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.MSELoss()
    sparsity_levels = [0.0, 0.5, 0.7, 0.8, 0.85, 0.9, 0.92, 0.95, 0.98]

    print(f"\nRunning experiment: {args.experiment}")
    print(f"Device: {device} | Epochs: {args.epochs} | Batch size: {args.batch_size}\n")

    if args.experiment == "superposition->robustness":
        results = run_experiment_32(sparsity_levels, args.epochs, args.batch_size, criterion, device)
    else:
        results = run_experiment_33(sparsity_levels, args.epochs, args.batch_size, criterion, device)

    print("\nFinal Results:")
    print(results)


if __name__ == "__main__":
    main()
