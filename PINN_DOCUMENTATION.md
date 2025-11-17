# Physics-Informed Neural Networks (PINNs) Implementation

## Overview

This implementation provides a Physics-Informed Neural Network (PINN) framework based on the approach described in Raissi et al. (2019) for solving partial differential equations (PDEs). The framework uses automatic differentiation to compute derivatives and enforces physical constraints through the loss function.

## Key Concepts from Raissi et al. (2019)

### 1. Neural Network Architecture
- **Input**: Spatial coordinates (x) and temporal coordinates (t)
- **Output**: Solution u(x, t)
- **Architecture**: Feedforward neural network with multiple hidden layers
- **Activation**: Tanh (default) or other activation functions

### 2. Automatic Differentiation
- Uses PyTorch's automatic differentiation to compute:
  - First derivatives: ∂u/∂x, ∂u/∂t
  - Second derivatives: ∂²u/∂x², ∂²u/∂t²
- This eliminates the need for finite difference approximations

### 3. Loss Function Components

The total loss function consists of three main components:

```
Loss_total = Loss_IC + Loss_BC + Loss_PDE
```

Where:
- **Loss_IC**: Loss from initial conditions (ensures u(x, 0) matches the initial condition)
- **Loss_BC**: Loss from boundary conditions (ensures boundary constraints are satisfied)
- **Loss_PDE**: Loss from PDE residual (ensures the solution satisfies the PDE)

### 4. Training Strategy
- Use collocation points throughout the domain to enforce the PDE
- Combine data-driven (IC/BC) and physics-driven (PDE) constraints
- Adam optimizer with learning rate scheduling

## Implemented PDEs

### 1. Heat Equation

**PDE**: ∂u/∂t = α * ∂²u/∂x²

**Problem Setup**:
- Domain: x ∈ [0, 1], t ∈ [0, 1]
- Initial condition: u(x, 0) = sin(πx)
- Boundary conditions: u(0, t) = u(1, t) = 0
- Analytical solution: u(x, t) = sin(πx) * exp(-απ²t)

**Files**:
- `pinn_model.py`: Core PINN implementation for Heat equation
- `train_heat_equation.py`: Training script with visualization

### 2. Wave Equation

**PDE**: ∂²u/∂t² = c² * ∂²u/∂x²

**Problem Setup**:
- Domain: x ∈ [0, 1], t ∈ [0, 1]
- Initial conditions: 
  - u(x, 0) = sin(πx)
  - u_t(x, 0) = 0
- Boundary conditions: u(0, t) = u(1, t) = 0
- Analytical solution: u(x, t) = sin(πx) * cos(πct)

**Files**:
- `wave_equation_pinn.py`: PINN implementation for Wave equation

## Usage

### Quick Test

```bash
conda activate pinns
python test_pinn.py
```

This runs a quick test with 500 epochs to verify the framework works correctly.

### Full Training (Heat Equation)

```bash
conda activate pinns
python train_heat_equation.py
```

This will:
1. Generate training data (IC, BC, and PDE collocation points)
2. Train the PINN for 10,000 epochs
3. Compute L2 error against analytical solution
4. Generate comprehensive visualizations
5. Save the trained model

### Training Parameters

You can modify the following parameters in the training scripts:

- **Network architecture**: `layers = [2, 50, 50, 50, 1]`
- **Thermal diffusivity**: `alpha = 0.1` (for Heat equation)
- **Wave speed**: `c = 1.0` (for Wave equation)
- **Training epochs**: `epochs = 10000`
- **Learning rate**: `lr = 0.001`
- **Number of collocation points**: `N_pde = 10000`

## Results and Validation

### L2 Error Tracking

The framework tracks the L2 error between predicted and exact solutions:

```
L2 Error = sqrt(mean((u_predicted - u_exact)²))
```

### Convergence Analysis

The training history includes:
- Total loss evolution
- Individual loss components (IC, BC, PDE)
- L2 error convergence
- Loss component comparison

### Visualizations

The framework generates:
1. **3D surface plots**: Exact vs predicted solutions
2. **Error plots**: Absolute error distribution
3. **Time slices**: Solution at different times
4. **Spatial slices**: Solution at different spatial points
5. **Training history**: Loss convergence plots

## Key Features

### 1. Automatic Differentiation
- Computes derivatives without finite differences
- Enables accurate PDE residual computation
- Works seamlessly with neural network gradients

### 2. Physics-Informed Loss
- Enforces physical constraints through the loss function
- Combines data and physics seamlessly
- No need for labeled data in the interior domain

### 3. Flexible Architecture
- Easy to modify network architecture
- Supports different activation functions
- GPU acceleration support

### 4. Validation Framework
- Comparison with analytical solutions
- L2 error tracking
- Comprehensive visualization tools

## File Structure

```
PINNS/
├── pinn_model.py              # Core PINN implementation (Heat equation)
├── wave_equation_pinn.py      # Wave equation PINN implementation
├── train_heat_equation.py     # Full training script for Heat equation
├── test_pinn.py               # Quick test script
├── verify_gpu.py              # GPU verification script
├── environment.yml            # Conda environment
└── README.md                  # Project documentation
```

## Next Steps

After successfully testing on basic PDEs:

1. **Phase 2**: Apply to more complex problems
   - Allen-Cahn equation
   - Navier-Stokes equations
   - Other nonlinear PDEs

2. **CAN-PINNs Implementation**
   - Implement Conservative Allen-Cahn Neural PINNs
   - Compare with baseline PINN results

3. **Advanced Features**
   - Adaptive sampling strategies
   - Multi-scale architectures
   - Transfer learning for parameterized PDEs

## References

Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378, 686-707.

## Troubleshooting

### High L2 Error
- Increase training epochs
- Increase number of collocation points
- Adjust learning rate
- Try different network architectures

### Training Instability
- Reduce learning rate
- Use learning rate scheduling
- Check data normalization
- Verify analytical solution is correct

### GPU Not Detected
- Run `python verify_gpu.py` to check GPU setup
- Ensure CUDA is properly installed
- Check PyTorch CUDA installation

## Performance Notes

- **GPU Acceleration**: The framework automatically uses GPU if available
- **Training Time**: Approximately 5-10 minutes for 10,000 epochs on T2000 GPU
- **Memory Usage**: Depends on number of collocation points (typically 1-2 GB)

## Contact

For questions or issues, refer to the project documentation or the research team.

