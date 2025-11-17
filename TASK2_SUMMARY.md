# Task 2: Numerical Experiments and Validation - Summary

## ✅ Implementation Complete

Successfully implemented **CAN-PINN** (Conservative Allen-Cahn Neural Physics-Informed Neural Network) with enhancements over standard PINN, and created comprehensive comparison framework for the Allen-Cahn equation.

## 📋 Implemented Features

### 1. ✅ Standard PINN for Allen-Cahn Equation

**File**: `allen_cahn_pinn.py` - `AllenCahnPINN` class

- **Automatic Differentiation**: Uses PyTorch autograd for all derivatives (∂u/∂t, ∂²u/∂x²)
- **Loss Function**: Combines IC, BC, and PDE residual losses
- **Architecture**: Configurable network (default: [2, 50, 50, 50, 1])
- **Training**: Adam optimizer with learning rate scheduling

### 2. ✅ CAN-PINN Implementation

**File**: `allen_cahn_pinn.py` - `CANAllenCahnPINN` class

**Key Enhancements**:

1. **Numerical Differentiation for Spatial Derivatives**:
   - Uses **central difference** for ∂²u/∂x²: `(u(x+h,t) - 2*u(x,t) + u(x-h,t)) / h²`
   - Automatic differentiation still used for ∂u/∂t
   - Handles boundary points with adjusted step sizes to preserve gradient flow

2. **Adaptive Loss Weighting**:
   - Dynamically adjusts weights for IC, BC, and PDE loss components
   - Based on gradient magnitudes of each loss term
   - Weights updated every 100 epochs during training
   - Clamped to prevent extreme values (0.1 to 10.0)

### 3. ✅ Comprehensive Training Script

**File**: `train_allen_cahn.py`

**Features**:
- Implements all test cases from Task 2
- Side-by-side comparison of PINN vs CAN-PINN
- Automatic visualization generation
- Performance metrics tracking (training time, final loss, convergence)

## 🧪 Test Cases Implemented

### Test Case 1: Basic PDE (Heat Equation)
- ✅ Already implemented in Task 1
- Validates framework with simple linear PDE

### Test Case 2: Varying Initial Conditions
- ✅ `u(x, 0) = sin(πx)` - Smooth initial condition
- ✅ `u(x, 0) = 1 for x > 0.5, 0 otherwise` - Step function (sharp interface)

### Test Case 3: Varying Diffusivity (ε)
- ✅ `ε = 0.01` - Sharp interfaces
- ✅ `ε = 0.05` - Smoother interfaces

### Test Case 4: Larger and Non-Uniform Grids
- ✅ Extended domain: `x ∈ [0, 2]`, `t ∈ [0, 2]`
- ✅ Non-uniform sampling: Denser near boundaries and initial time

## 📊 Visualization Features

The comparison visualization includes:

1. **3D Surface Plots**:
   - PINN solution
   - CAN-PINN solution
   - Absolute difference between them

2. **2D Solution Profiles**:
   - Solution at different times (t = 0, 0.25, 0.5, 0.75, 1.0)
   - Solution at different spatial points (x = 0.25, 0.5, 0.75)
   - Direct comparison between PINN and CAN-PINN

3. **Training Analysis**:
   - Loss convergence comparison (log scale)
   - PDE loss component comparison
   - Adaptive loss weights evolution (CAN-PINN)

4. **Error Distribution**:
   - Histogram of absolute errors

## 🔑 Key Technical Details

### Numerical Differentiation
- **Step size (h)**: Default 0.01 (configurable)
- **Boundary handling**: Adjusts step size near boundaries instead of clamping
- **Gradient preservation**: Uses `torch.where` to maintain gradient flow

### Adaptive Loss Weighting
- **Update frequency**: Every 100 epochs
- **Weight calculation**: Inverse proportional to gradient magnitudes
- **Normalization**: Ensures balanced contribution of all loss components
- **Clamping**: Prevents weights from becoming too extreme (0.1 to 10.0)

### Training Strategy
- **Optimizer**: Adam with initial learning rate 0.001
- **Scheduler**: ReduceLROnPlateau (reduces LR by 0.5 when loss plateaus)
- **Default epochs**: 10,000 (configurable)
- **Print frequency**: Every 1000 epochs (configurable)

## 📈 Expected Results

Based on CAN-PINN design principles:

1. **Faster Convergence**: Adaptive loss weighting should help balance training
2. **Better Accuracy**: Numerical differentiation may provide more stable gradients for sharp interfaces
3. **Improved Stability**: Adaptive weights prevent one loss component from dominating

## 🚀 Usage

### Quick Test
```bash
conda activate pinns
python test_allen_cahn.py
```

### Full Training (All Test Cases)
```bash
conda activate pinns
python train_allen_cahn.py
```

### Individual Test Case
```python
from train_allen_cahn import run_test_case

# Test Case 2: sin(πx) initial condition
results = run_test_case(2, epsilon=0.01, ic_type='sin', epochs=10000)

# Test Case 2: Step function initial condition
results = run_test_case(2, epsilon=0.01, ic_type='step', epochs=10000)

# Test Case 3: Different diffusivity
results = run_test_case(3, epsilon=0.01, ic_type='sin', epochs=10000)
results = run_test_case(3, epsilon=0.05, ic_type='sin', epochs=10000)

# Test Case 4: Larger domain with non-uniform grids
results = run_test_case(4, epsilon=0.01, ic_type='sin', epochs=10000)
```

## 📝 Files Created

1. **`allen_cahn_pinn.py`**: Core implementations (PINN and CAN-PINN)
2. **`train_allen_cahn.py`**: Comprehensive training and comparison script
3. **`test_allen_cahn.py`**: Quick test script
4. **`TASK2_SUMMARY.md`**: This document

## 🔍 Comparison Metrics

The framework tracks:

1. **Convergence Rate**: Loss decrease over epochs
2. **Final Loss**: Total loss at end of training
3. **Component Losses**: Individual IC, BC, and PDE losses
4. **Training Time**: Time taken for training
5. **Adaptive Weights**: Evolution of loss weights (CAN-PINN only)

## ⚠️ Notes

1. **No Exact Solution**: Allen-Cahn equation generally doesn't have analytical solutions for arbitrary initial conditions. Comparison is based on:
   - Loss convergence
   - Solution smoothness and physical behavior
   - Comparison between PINN and CAN-PINN

2. **Numerical Differentiation**: The step size `h` is a hyperparameter. May need tuning for different problems.

3. **Adaptive Weights**: The weighting mechanism is designed to balance training. Initial weights may need adjustment for specific problems.

## 🎯 Next Steps

1. **Run Full Experiments**: Execute all test cases with full training epochs
2. **Analyze Results**: Compare convergence rates and final losses
3. **Tune Hyperparameters**: Adjust `h`, learning rate, and network architecture if needed
4. **Phase 3**: Implement further enhancements (Fourier features, adaptive sampling, curriculum learning)

## 📚 References

- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378, 686-707.

---

**Status**: ✅ Implementation Complete - Ready for experiments

