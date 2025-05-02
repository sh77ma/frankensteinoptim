
"""
PyTorch implementation of the Frankenstein Optimizer.

This optimizer combines techniques from various adaptive gradient-based
optimizers to potentially achieve faster convergence and better generalization.
The implementation closely follows Algorithm 1 presented in the source paper.
"""

import torch
import torch.optim as optim
import math

class Frankenstein(optim.Optimizer):
    r"""Implements the Frankenstein optimizer algorithm.

    This implementation is based on the paper:
    "Frankenstein Optimizer: Harnessing the Potential by Revisiting Optimization Tricks"
    Authors: Chia-Wei Hsu, Nien-Ti Tsou, Yu-Cheng Chen, Yang Jeong Park, Ju Li
    arXiv: 2503.02147v1 [cs.LG] 4 Mar 2025 (Note: Year corrected to likely intended 2024/2025, adjust if needed)
    Link: https://arxiv.org/abs/2503.02147 (Link may become active later)

    Arguments:
        params (iterable): Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float, optional): Learning rate (alpha_t in the paper's Algorithm 1
            and equations). (default: 1e-3)
        beta1_default (float, optional): Default coefficient used only to derive the
            reference factor for clipping beta1 (specifically, `1 - beta1_default`).
            The paper uses beta1_default=0.9, implying a factor of 0.1.
            This parameter is *NOT* directly used as the momentum coefficient beta1.
            (default: 0.9)
        eps (float, optional): Term added to the denominator and squared gradients
            (chi_t) to improve numerical stability. (default: 1e-8)
        clip_beta1_lower (float, optional): Lower bound for the *term* used to
            calculate adaptive beta1_t (`0.1 * sqrt(alpha_t / alpha_0)` in Alg 1).
            Corresponds to the lower clip bound in Alg 1, line 6. (default: 0.05)
        clip_beta1_upper (float, optional): Upper bound for the *term* used to
            calculate adaptive beta1_t (`0.1 * sqrt(alpha_t / alpha_0)` in Alg 1).
            Corresponds to the upper clip bound in Alg 1, line 6. (default: 0.99)
        clip_rho_lower (float, optional): Lower bound for the input to the log
            when calculating rho (parameter p in Alg 1, line 9).
            Should be > 0. Defaults to `exp(0.8)` as used in Alg 1's Clip.
            (default: math.exp(0.8))
        clip_rho_upper (float, optional): Upper bound for the input to the log
            when calculating rho (parameter p in Alg 1, line 9).
            Defaults to `exp(1.05)` as used in Alg 1's Clip.
            (default: math.exp(1.05))
        beta1_ref_lr (float, optional): Reference learning rate (alpha_0 in Eq. 3,
            used as 10^-3 in Alg 1, line 6) for adapting beta1. (default: 1e-3)

    Example:
        >>> optimizer = Frankenstein(model.parameters(), lr=0.001)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    """

    def __init__(self, params, lr=1e-3, beta1_default=0.9, eps=1e-8,
                 clip_beta1_lower=0.05, clip_beta1_upper=0.99,
                 clip_rho_lower=math.exp(0.8), clip_rho_upper=math.exp(1.05),
                 beta1_ref_lr = 1e-3):

        # --- Input Validation ---
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        # beta1_default is only used to imply the 0.1 factor, check its range anyway
        if not 0.0 <= beta1_default < 1.0:
             raise ValueError("Invalid beta1_default value: {}".format(beta1_default))
        # Validate the term used for beta1 clipping, not beta1 itself
        if not 0.0 <= clip_beta1_lower <= clip_beta1_upper < 1.0:
            raise ValueError(f"Invalid beta1 clipping bounds for the term: 0 <= {clip_beta1_lower} <= {clip_beta1_upper} < 1 required")
        # Rho clipping bounds must be positive for the log argument
        if not 0.0 < clip_rho_lower <= clip_rho_upper:
             raise ValueError(f"Invalid rho clipping bounds: 0 < {clip_rho_lower} <= {clip_rho_upper} required")
        if not 0.0 < beta1_ref_lr:
             raise ValueError("Invalid beta1_ref_lr value: {}".format(beta1_ref_lr))
        # --- End Input Validation ---

        defaults = dict(lr=lr, beta1_default=beta1_default, eps=eps,
                        clip_beta1_lower=clip_beta1_lower,
                        clip_beta1_upper=clip_beta1_upper,
                        clip_rho_lower=clip_rho_lower,
                        clip_rho_upper=clip_rho_upper,
                        beta1_ref_lr=beta1_ref_lr)
        super(Frankenstein, self).__init__(params, defaults)
        print("Initializing Frankenstein optimizer...")

    def __setstate__(self, state):
        """Loads optimizer state."""
        super(Frankenstein, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss. This is required for some optimizers like LBFGS,
                but optional for Frankenstein.
        """
        loss = None
        if closure is not None:
            # Ensure gradients are enabled for the closure evaluation
            with torch.enable_grad():
                loss = closure()

        # Iterate over parameter groups (e.g., different LRs for different layers)
        for group in self.param_groups:
            # Retrieve group-specific hyperparameters
            lr = group['lr']
            beta1_default = group['beta1_default'] # Only used for the 0.1 factor
            eps = group['eps']
            clip_beta1_lower = group['clip_beta1_lower']
            clip_beta1_upper = group['clip_beta1_upper']
            clip_rho_lower = group['clip_rho_lower']
            clip_rho_upper = group['clip_rho_upper']
            beta1_ref_lr = group['beta1_ref_lr']

            # --- Calculate adaptive beta1 (scalar for the group, beta1_t) ---
            # Based on Alg 1, line 6: β₁,ₜ ← 1 - Clip(0.1 * sqrt(αₜ / 10⁻³), 0.05, 0.99)
            # The 0.1 factor is derived from (1 - beta1_default)
            ref_factor = 1.0 - beta1_default # Typically 0.1
            # Use max with eps for numerical stability in sqrt and division
            alpha_t_safe = max(lr, eps)
            alpha_0_safe = max(beta1_ref_lr, eps) # alpha_0 = beta1_ref_lr
            # Argument inside Clip function
            clip_arg = ref_factor * math.sqrt(alpha_t_safe / alpha_0_safe)
            # Apply Clip - result is a scalar for the whole group
            clipped_value = min(max(clip_arg, clip_beta1_lower), clip_beta1_upper)
            # Final adaptive momentum coefficient beta1_t (scalar)
            beta1_t = 1.0 - clipped_value
            # --- End adaptive beta1 calculation ---

            # Iterate over parameters within the group
            for p in group['params']:
                # Skip parameters without gradients
                if p.grad is None:
                    continue
                grad = p.grad
                # Raise error for sparse gradients (not supported)
                if grad.is_sparse:
                    raise RuntimeError('Frankenstein optimizer does not support sparse gradients')

                # Access parameter-specific state
                state = self.state[p]

                # State initialization (for the first step)
                if len(state) == 0:
                    state['step'] = 0
                    # exp_avg: First moment vector (momentum term, m_t in Alg 1)
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # exp_avg_sq: Second moment EMA (v_t in paper notation, used for beta2_t)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # exp_avg_sq_max: Max observed squared gradient (v_hat_t in Alg 1, used in denom)
                    state['exp_avg_sq_max'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # chi_prev: Previous squared gradient + eps (χ_{t-1} in Alg 1)
                    # Initialize with eps for stability in first beta2_t calculation
                    state['chi_prev'] = torch.full_like(p, eps, memory_format=torch.preserve_format)

                # Retrieve state variables for the current step
                exp_avg = state['exp_avg']          # m_{t-1}
                exp_avg_sq = state['exp_avg_sq']    # v_{t-1}
                exp_avg_sq_max = state['exp_avg_sq_max'] # v̂_{t-1}
                chi_prev = state['chi_prev']        # χ_{t-1}

                # Increment step counter
                state['step'] += 1
                # step = state['step'] # Available if bias correction were needed

                # --- Start Calculations based on Algorithm 1 ---

                # Alg 1, Line 7: Calculate Non-linear Misalignment Factor P
                # P ← arccos(tanh(mₜ₋₁ ⋅ gₜ)) / π (using element-wise product)
                tanh_val = torch.tanh(exp_avg * grad)
                # Clamp input to acos to avoid NaN from values slightly outside [-1, 1] due to precision
                acos_input = torch.clamp(tanh_val, -1.0 + eps, 1.0 - eps)
                p_factor = torch.acos(acos_input) / math.pi # Result is a Tensor P

                # Alg 1, Line 8: Calculate current squared gradient term chi_t
                # χₜ ← gₜ² + ε
                chi_t = grad.pow(2).add(eps) # Use add, not add_ to avoid modifying grad if used elsewhere

                # Alg 1, Line 8: Update maximum observed squared gradient v_hat_t
                # v̂ₜ ← max(v̂ₜ₋₁, χₜ)
                # Update exp_avg_sq_max (v̂ₜ) in-place for efficiency
                torch.maximum(exp_avg_sq_max, chi_t, out=exp_avg_sq_max)

                # Calculate denominator for updates: sqrt(v̂ₜ) + ε
                denom = exp_avg_sq_max.sqrt().add(eps) # Use add, not add_

                # Alg 1, Line 9: Calculate adaptive coefficient rho (p in Alg 1)
                # ρ ← log(Clip(e¹ + sqrt(χₜ) + 0.5 - P, e⁰.⁸, e¹.⁰⁵))
                rho_input = math.e + chi_t.sqrt() + 0.5 - p_factor
                # Clip the input using pre-defined bounds
                rho_clipped = torch.clamp(rho_input, min=clip_rho_lower, max=clip_rho_upper)
                rho = torch.log(rho_clipped) # Result is a Tensor ρ

                # Alg 1, Line 10: Calculate acceleration factor xi (ξ in Alg 1)
                # ξ ← (1 + e⁻⁰.⁵) / (1 + e⁻|χₜ₋₁ - P|)
                xi_denom_exp_input = -(chi_prev - p_factor).abs()
                # Numerator is a constant scalar
                xi_numerator = 1.0 + math.exp(-0.5)
                # Denominator is 1 + exp(tensor)
                xi_denominator = 1.0 + torch.exp(xi_denom_exp_input)
                xi = xi_numerator / xi_denominator # Result is a Tensor ξ

                # Alg 1, Line 11: Update first moment m_t
                # mₜ ← ρ * β₁,ₜ * mₜ₋₁ - αₜ * gₜ / sqrt(v̂ₜ)
                # Note: beta1_t is scalar, rho is tensor. grad_update = g_t / denom
                grad_update_term = grad / denom
                # Update exp_avg (mₜ = m_{t-1}) in-place
                exp_avg.mul_(rho * beta1_t).add_(grad_update_term, alpha=-lr) # m_t = rho*beta1_t*m_{t-1} - lr * (g_t / denom)

                # Alg 1, Line 12: Update parameters theta_t
                # θₜ ← θₜ₋₁ + β₁,ₜ * mₜ - αₜ * gₜ * ξ / sqrt(v̂ₜ)
                # We use the m_t we just calculated (now stored in exp_avg)
                # Calculate the second term involving xi: αₜ * gₜ * ξ / sqrt(v̂ₜ)
                xi_update_term = (grad * xi) / denom
                # Perform the parameter update: p = p + beta1_t * m_t - lr * xi_update_term
                # Update p (θ_{t-1}) in-place
                p.add_(exp_avg * beta1_t).add_(xi_update_term, alpha=-lr) # p = p + beta1_t*m_t - lr*(g_t*xi/denom)

                # --- Update second moment EMA v_t ---
                # Alg 1, Line 12 (second part): Calculate adaptive beta2_t
                # β₂,ₜ ← 1 - |χₜ / χₜ₋₁ - |0.5 - P||
                safe_chi_prev = chi_prev # Already includes eps from previous step or initialization
                chi_ratio = chi_t / safe_chi_prev
                abs_p_term = (0.5 - p_factor).abs()
                beta2_t_abs_term = (chi_ratio - abs_p_term).abs()
                # Final beta2_t tensor
                beta2_t = 1.0 - beta2_t_abs_term
                # Clamp beta2_t to [0, ~1) for stability of EMA update
                beta2_t = torch.clamp(beta2_t, min=0.0, max=0.99999) # Tensor β₂,ₜ

                # Alg 1, Line 13: Update second moment EMA v_t
                # vₜ ← β₂,ₜ * vₜ₋₁ + (1 - β₂,ₜ) * χₜ
                # Update exp_avg_sq (v_{t-1}) in-place using the just computed beta2_t
                exp_avg_sq.mul_(beta2_t).addcmul_(1.0 - beta2_t, chi_t, value=1.0) # v_t = beta2_t*v_{t-1} + (1-beta2_t)*chi_t
                # Alternative: exp_avg_sq.mul_(beta2_t).add_((1.0 - beta2_t) * chi_t) # Should be equivalent

                # --- Store state for next iteration ---
                # exp_avg, exp_avg_sq, exp_avg_sq_max were updated in-place
                # Store the current chi_t (which includes eps) as chi_prev for the next step
                state['chi_prev'].copy_(chi_t)
                # --- End Calculations ---

        # Return loss if closure was provided
        return loss

