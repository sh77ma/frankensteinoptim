# Frankenstein Optimizer - PyTorch Implementation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository provides a PyTorch implementation of the **Frankenstein Optimizer**, based on the paper:

> **Frankenstein Optimizer: Harnessing the Potential by Revisiting Optimization Tricks**
>
> *Authors: Chia-Wei Hsu, Nien-Ti Tsou, Yu-Cheng Chen, Yang Jeong Park, Ju Li*
>
> *arXiv: 2503.02147v1 [cs.LG] 4 Mar 2025*
>
> *Link: [https://arxiv.org/abs/2503.02147](https://arxiv.org/abs/2503.02147)* 

**Disclaimer:** This is an unofficial implementation created based on Algorithm 1 described in the paper. It aims for functional equivalence but may differ in minor details or optimizations.

---

## Overview
A while ago, I came across an article discussing the Frankenstein Optimizer.The Frankenstein optimizer aims to combine the advantages of various adaptive gradient-based methods. It dynamically adjusts its internal parameters based on the current state of the optimization process, potentially leading to faster convergence and improved generalization compared to optimizers like Adam or SGD in certain scenarios. The article did not provide the original code for the optimizer, so I developed my own implementation based on the description of the algorithm in the article. While my implementation may not be perfect, it has several advantages over AdamW and other similar optimizers, including faster convergence and better generalization due to its learnable and adjustable parameters.
My implementation is available for use and I invite users to report any bugs or issues they may encounter. I have successfully used this implementation in certain ViT-like models for computer vision applications. However, it should be noted that the search for local minima and resulting landscapes differ from those of AdamW and SophiaG.




---

## Key Features (based on the paper)

Dynamically adapts its internal coefficients during training:

*   **Adaptive First Momentum Coefficient (`β₁`):** Adjusts based on the learning rate, unlike the fixed `β₁` in Adam.
*   **Dynamic Second Moment EMA (`β₂`):** Adapts based on the ratio of current to past squared gradients and the alignment between momentum and current gradient.
*   **Max-based Second Moment Normalization (`v̂`):** Uses the maximum observed squared gradient (similar to AMSGrad) for parameter update normalization, potentially enhancing stability.

Incorporates novel factors for fine-grained control:

*   **Nonlinear Misalignment Factor (`P`):** Quantifies the misalignment between the previous momentum and current gradient.
*   **Adaptive Coefficient (`ρ`):** Modulates the momentum update based on gradient magnitude and the misalignment factor `P`.
*   **Acceleration Factor (`ξ`):** Adjusts the final parameter update, potentially accelerating convergence in consistent directions and stabilizing during rapid changes.

---

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/sh77ma/frankensteinoptim
    cd frankensteinoptim
    ```

2.  **Ensure PyTorch is installed:**
    Requires `torch >= 1.8` (might work with earlier versions). See [pytorch.org](https://pytorch.org/) for installation instructions.

3.  **Import the optimizer:**
    Place the `frankenstein_optimizer.py` file in your project directory or ensure it's in your Python path.

---

## Usage Example

Import and use `Frankenstein` like any standard PyTorch optimizer.

```python
import torch
import torch.nn as nn
from frankenstein_optimizer import Frankenstein # Make sure the file is accessible

# 1. Define your model, loss, and data
model = nn.Linear(10, 1) # Example
criterion = nn.MSELoss()
# data_loader = ... # Your DataLoader setup

# 2. Initialize the optimizer
#    Use default parameters:
#    optimizer = Frankenstein(model.parameters())
#    Or customize:
optimizer = Frankenstein(model.parameters(), lr=0.001, eps=1e-7)

# 3. Standard Training Loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    # for inputs, targets in data_loader: # Loop through your data
    # Simulate some data for example:
    inputs = torch.randn(4, 10)
    targets = torch.randn(4, 1)

    # Standard PyTorch steps
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Add validation, saving, etc. as needed
    # ...
```
## Parameters

The `Frankenstein` optimizer class accepts the following arguments during initialization:

*   `params` (`iterable`):
    Iterable of parameters (usually `model.parameters()`) to optimize or a list of dicts defining separate parameter groups (e.g., for different learning rates).

*   `lr` (`float`, *optional*):
    Learning rate (denoted as αₜ in the paper's algorithm). Controls the overall step size.
    *Default: `1e-3`*

*   `beta1_default` (`float`, *optional*):
    A coefficient used *only* to determine the reference factor (typically 0.1) for the adaptive β₁ calculation, via the formula `1 - beta1_default`. It is **not** the direct momentum coefficient β₁.
    *Default: `0.9` (implying a factor of 0.1)*

*   `eps` (`float`, *optional*):
    A small constant added to denominators and squared gradients (χₜ) to prevent division by zero and improve numerical stability.
    *Default: `1e-8`*

*   `clip_beta1_lower` (`float`, *optional*):
    The lower bound applied to the term `(1 - beta1_default) * sqrt(αₜ / α₀)` *before* calculating the final adaptive β₁. Corresponds to the lower clip value in Algorithm 1, line 6.
    *Default: `0.05`*

*   `clip_beta1_upper` (`float`, *optional*):
    The upper bound applied to the term `(1 - beta1_default) * sqrt(αₜ / α₀)` *before* calculating the final adaptive β₁. Corresponds to the upper clip value in Algorithm 1, line 6. Must be `< 1.0`.
    *Default: `0.99`*

*   `clip_rho_lower` (`float`, *optional*):
    The lower bound applied to the *input* of the `log` function when calculating the adaptive coefficient ρ (parameter `p` in Algorithm 1, line 9). Must be strictly greater than 0. Corresponds to the lower clip value `exp(0.8)` in Algorithm 1.
    *Default: `math.exp(0.8)`*

*   `clip_rho_upper` (`float`, *optional*):
    The upper bound applied to the *input* of the `log` function when calculating the adaptive coefficient ρ. Corresponds to the upper clip value `exp(1.05)` in Algorithm 1.
    *Default: `math.exp(1.05)`*

*   `beta1_ref_lr` (`float`, *optional*):
    The reference learning rate (α₀ in Eq. 3, value `10^-3` used in Algorithm 1, line 6) used in the adaptive β₁ calculation.
    *Default: `1e-3`*

---

## Citation

If you use this optimizer implementation or find the original paper useful in your research, please consider citing the paper:

```bibtex
@misc{hsu2025frankenstein,
      title={Frankenstein Optimizer: Harnessing the Potential by Revisiting Optimization Tricks},
      author={Chia-Wei Hsu and Nien-Ti Tsou and Yu-Cheng Chen and Yang Jeong Park and Ju Li},
      year={2025},
      eprint={2503.02147},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

License
This specific implementation is released under the MIT License. Please see the LICENSE file for details.

Acknowledgements
Deep gratitude to the authors of the original paper for developing the Frankenstein optimizer algorithm. This implementation is an effort to faithfully reproduce Algorithm 1 from their work for the benefit of the community.
