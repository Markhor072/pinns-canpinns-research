# Improved CAN-PINN: Implementation Guide

## Overview

This document describes the improvements made to the CAN-PINN implementation to address the issues identified in the initial experiments where CAN-PINN performed worse than standard PINN.

## Key Improvements

### 1. Adaptive Step Size (h) for Numerical Differentiation

**Problem**: Fixed `h=0.01` was not optimal for all sampling densities and could cause numerical errors.

**Solution**: 
- Compute adaptive `h` based on median spacing of training data
- `h = 2.0 * median(dx)` where `dx` is the spacing between adjacent points
- Clamped between `0.001` and `0.1` for stability
- Updated automatically from training data

**Implementation**: `update_h_from_data()` method

### 2. Improved Boundary Handling

**Problem**: Boundary points using one-sided differences introduced bias and gradient flow issues.

**Solution**:
- Use symmetric extension (reflecting boundary) for points near boundaries
- Left boundary: Use `x + 2h` instead of `x - h` (symmetric extension)
- Right boundary: Use `x - 2h` instead of `x + h` (symmetric extension)
- Maintains gradient flow better than clamping

**Implementation**: Enhanced `pde_residual()` method with boundary masks

### 3. Uncertainty-Based Loss Weighting

**Problem**: Previous adaptive weighting based on gradient magnitudes caused weights to saturate at maximum values (10.0), leading to poor training dynamics.

**Solution**:
- Use learned uncertainty weights (homoscedastic uncertainty)
- Learnable parameters: `log_var_ic`, `log_var_bc`, `log_var_pde`
- Loss formulation:
  ```
  L = (1/2) * (1/σ²) * error² + (1/2) * log(σ²)
  ```
- Weights adapt naturally during training without hard constraints
- Prevents saturation and provides better balance

**Implementation**: `use_uncertainty_weights=True` with learnable log-variance parameters

### 4. Gradient Penalty Regularization

**Problem**: Sharp interfaces in Allen-Cahn equation can cause oscillations and instabilities.

**Solution**:
- Add gradient penalty: `λ * mean(||∇u||²)`
- Promotes smoother solutions
- Weight `λ = 1e-5` (configurable)
- Helps stabilize training, especially for sharp interfaces

**Implementation**: `gradient_penalty_weight` parameter in loss function

### 5. L-BFGS Fine-Tuning

**Problem**: Adam optimizer may not converge to optimal solution, especially for physics constraints.

**Solution**:
- Two-phase training:
  1. **Phase 1**: Adam optimizer (fast initial convergence)
  2. **Phase 2**: L-BFGS optimizer (fine-tuning, better for physics constraints)
- L-BFGS is more effective for minimizing PDE residuals
- Typically use 9000 Adam epochs + 1000 L-BFGS iterations

**Implementation**: `use_lbfgs=True`, `lbfgs_epochs=1000`

### 6. Residual-Based Adaptive Sampling (RBAS)

**Problem**: Uniform sampling may miss important regions (e.g., sharp interfaces, boundaries).

**Solution**:
- Resample collocation points based on PDE residual magnitude
- Keep points with highest residuals (most informative)
- Resample new points near high-residual regions
- Oversample near boundaries for sharp interfaces
- Resample every N epochs (default: 1000)

**Implementation**: `ResidualAdaptiveSampler` class

### 7. Fourier Feature Encoding (Optional)

**Problem**: Standard MLPs may struggle with high-frequency features (sharp interfaces).

**Solution**:
- Add Fourier feature encoding: `γ(x) = [sin(2πBx), cos(2πBx), x]`
- Random frequency matrix `B` sampled from `N(0, γ²)`
- Helps network capture high-frequency patterns
- Parameter `γ` controls frequency range (default: 10.0)

**Implementation**: `fourier_features=True`, `gamma=10.0`

## Usage

### Basic Usage

```python
from allen_cahn_pinn_improved import ImprovedCANAllenCahnPINN

# Initialize model
model = ImprovedCANAllenCahnPINN(
    epsilon=0.01,
    layers=[2, 50, 50, 50, 1],
    h_adaptive=True,              # Adaptive step size
    use_uncertainty_weights=True, # Uncertainty weighting
    gradient_penalty_weight=1e-5, # Gradient penalty
    fourier_features=False,       # Fourier features (optional)
    gamma=10.0
)

# Train
history = model.train(
    x_ic, t_ic, u_ic,
    x_bc, t_bc, u_bc,
    x_pde, t_pde,
    epochs=10000,
    lr=0.001,
    use_lbfgs=True,      # Use L-BFGS fine-tuning
    lbfgs_epochs=1000
)
```

### With Residual-Based Adaptive Sampling

```python
from residual_adaptive_sampling import ResidualAdaptiveSampler

# Initialize sampler
sampler = ResidualAdaptiveSampler(
    initial_N=20000,
    resample_frequency=1000,
    resample_fraction=0.2,
    keep_best_fraction=0.8
)

# During training, resample periodically
for epoch in range(0, epochs, 1000):
    if epoch % 1000 == 0 and epoch > 0:
        x_pde, t_pde = sampler.adaptive_sampling_step(
            model, x_pde, t_pde, epoch,
            x_min=0.0, x_max=1.0,
            t_min=0.0, t_max=1.0
        )
```

### Full Training Script

```bash
# Run improved CAN-PINN training
python train_improved_allen_cahn.py

# Run single test case
python run_single_test.py --test_case 2 --epsilon 0.01 --ic_type sin --epochs 10000
```

## Expected Improvements

### 1. Better Convergence
- Uncertainty weights prevent saturation
- L-BFGS fine-tuning improves final accuracy
- Adaptive sampling focuses on important regions

### 2. More Stable Training
- Gradient penalty reduces oscillations
- Better boundary handling prevents gradient issues
- Adaptive h reduces numerical errors

### 3. Better Accuracy
- All improvements work together to improve solution quality
- Especially beneficial for sharp interfaces (step function IC, small ε)

## Comparison with Previous Results

### Previous CAN-PINN Issues:
- ❌ Loss plateaued at high values (~0.08)
- ❌ Weights saturated at maximum (10.0)
- ❌ Poor performance on step function IC
- ❌ Worse than standard PINN

### Improved CAN-PINN Expected:
- ✅ Better convergence (lower final loss)
- ✅ Stable weight evolution
- ✅ Better handling of sharp interfaces
- ✅ Comparable or better than standard PINN

## Hyperparameter Tuning

### Key Hyperparameters:

1. **Gradient Penalty Weight**: `1e-5` to `1e-4`
   - Too small: No effect
   - Too large: Over-smoothing

2. **Uncertainty Weights**: Automatically learned
   - Initial values: `log_var = 0` (weight = 0.5)
   - Adapts during training

3. **L-BFGS Epochs**: `500` to `2000`
   - More epochs: Better fine-tuning but slower
   - Balance between accuracy and speed

4. **Adaptive Sampling Frequency**: `500` to `2000` epochs
   - More frequent: Better adaptation but more overhead
   - Less frequent: Less adaptive but faster

5. **Fourier Features Gamma**: `5.0` to `20.0`
   - Smaller: Lower frequencies
   - Larger: Higher frequencies (better for sharp interfaces)

## Next Steps

1. **Run Experiments**: Test improved CAN-PINN on all test cases
2. **Compare Results**: Compare with standard PINN and original CAN-PINN
3. **Tune Hyperparameters**: Optimize for each test case
4. **Analyze**: Understand which improvements contribute most
5. **Publish**: Document results and improvements

## Files

- `allen_cahn_pinn_improved.py`: Improved CAN-PINN implementation
- `residual_adaptive_sampling.py`: Adaptive sampling implementation
- `train_improved_allen_cahn.py`: Training script with comparisons
- `test_improved.py`: Quick test script
- `IMPROVEMENTS_GUIDE.md`: This document

## References

1. **Uncertainty Weighting**: Kendall et al. "Multi-Task Learning Using Uncertainty to Weigh Losses"
2. **Fourier Features**: Tancik et al. "Fourier Features Let Networks Learn High Frequency Functions"
3. **Adaptive Sampling**: Daw et al. "Physics-Informed Neural Networks with Adaptive Sampling"
4. **L-BFGS for PINNs**: Wang et al. "When and Why PINNs Fail to Train"

