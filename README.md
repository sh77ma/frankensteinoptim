# Frankenstein Optimizer - PyTorch Implementation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository provides a PyTorch implementation of the **Frankenstein Optimizer**, based on the paper:

**Frankenstein Optimizer: Harnessing the Potential by Revisiting Optimization Tricks**
*Authors: Chia-Wei Hsu, Nien-Ti Tsou, Yu-Cheng Chen, Yang Jeong Park, Ju Li*
*arXiv: 2503.02147v1 [cs.LG] 4 Mar 2025*
*Link: [https://arxiv.org/abs/2503.02147](https://arxiv.org/abs/2503.02147)* (Note: Link might become active later)

**Disclaimer:** This is an unofficial implementation created based on the algorithm described in the paper.

## Overview

The Frankenstein optimizer aims to combine the advantages of various adaptive gradient-based methods. It dynamically adjusts its internal parameters based on the current state of the optimization process, potentially leading to faster convergence and improved generalization compared to optimizers like Adam or SGD in certain scenarios.

## Key Features (based on the paper)

*   **Adaptive First Momentum Coefficient (β₁):** Dynamically adjusts based on the learning rate, unlike the fixed β₁ in Adam.
*   **Dynamic Second Moment EMA (β₂):** The coefficient β₂ for the exponential moving average of squared gradients adapts based on the ratio of current to past squared gradients and the alignment between momentum and current gradient.
*   **Max-based Second Moment for Normalization (v̂):** Uses the maximum observed squared gradient (`v̂`, similar to AMSGrad) in the denominator for parameter updates, potentially offering more stability.
*   **Nonlinear Misalignment Factor (P):** Quantifies the misalignment between the previous momentum direction and the current gradient direction.
*   **Adaptive Coefficient (ρ):** Modulates the momentum update based on gradient magnitude and the misalignment factor `P`.
*   **Acceleration Factor (ξ):** Adjusts the final parameter update based on gradient magnitude and the misalignment factor `P`, potentially accelerating convergence in consistent directions and stabilizing during rapid gradient changes.

## Installation

1.  Clone this repository:
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```
2.  Ensure you have PyTorch installed (`torch >= 1.8` recommended, though might work with earlier versions).
3.  Place the `frankenstein_optimizer.py` file in your project directory or a location accessible by your Python path.

## Usage

Import the optimizer and use it like any standard PyTorch optimizer:

```python
import torch
import torch.nn as nn
from frankenstein_optimizer import Frankenstein # Assuming the file is in the same directory or accessible

# Define your model, loss function, dataloader
model = nn.Linear(10, 1) # Example model
criterion = nn.MSELoss()
data_loader = ... # Your DataLoader

# Initialize the optimizer
# Using default parameters:
# optimizer = Frankenstein(model.parameters())
# Or with custom parameters:
optimizer = Frankenstein(model.parameters(), lr=0.001, eps=1e-7)

# Example training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for inputs, targets in data_loader:
        # inputs, targets = inputs.to(device), targets.to(device) # Move data to appropriate device

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()

        # Optimizer step
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Optional: Validation loop, saving checkpoints, etc.
    # ...

Parameters
The Frankenstein optimizer accepts the following parameters during initialization:

params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.

lr (float, optional): Learning rate (αₜ in the paper). Default: 1e-3.

beta1_default (float, optional): Used only to derive the 0.1 factor for β₁ calculation (as 1 - beta1_default). Not the actual momentum coefficient. Default: 0.9.

eps (float, optional): Small term added for numerical stability. Default: 1e-8.

clip_beta1_lower (float, optional): Lower clip bound for the term 0.1 * sqrt(αₜ / α₀) used in β₁ calculation. Default: 0.05.

clip_beta1_upper (float, optional): Upper clip bound for the term 0.1 * sqrt(αₜ / α₀) used in β₁ calculation. Default: 0.99.

clip_rho_lower (float, optional): Lower clip bound for the input to log in the calculation of ρ. Must be > 0. Default: math.exp(0.8).

clip_rho_upper (float, optional): Upper clip bound for the input to log in the calculation of ρ. Default: math.exp(1.05).

beta1_ref_lr (float, optional): Reference learning rate (α₀) used for β₁ adaptation. Default: 1e-3.

Citation
If you use this optimizer or find the paper useful, please cite the original work:
@misc{hsu2025frankenstein,
      title={Frankenstein Optimizer: Harnessing the Potential by Revisiting Optimization Tricks},
      author={Chia-Wei Hsu and Nien-Ti Tsou and Yu-Cheng Chen and Yang Jeong Park and Ju Li},
      year={2025},
      eprint={2503.02147},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}


License
This specific implementation is released under the MIT License.

Acknowledgements
Credit goes to the authors of the original paper for developing the Frankenstein optimizer algorithm. This implementation aims to faithfully reproduce Algorithm 1 from their work.
