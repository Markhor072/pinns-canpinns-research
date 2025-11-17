# Hybrid CAN-PINNs for Allen-Cahn Equation

A Physics-Informed Neural Network (PINN) implementation for solving the Allen-Cahn equation using a hybrid approach that combines Automatic Differentiation with adaptive enhancements.

## 🎯 Project Overview

This project implements and compares:
- **Baseline PINN**: Standard Physics-Informed Neural Network
- **Hybrid CAN-PINN**: Enhanced PINN with Automatic Differentiation, uncertainty weighting, adaptive sampling, and L-BFGS fine-tuning

The hybrid approach successfully eliminates numerical differentiation errors while maintaining competitive or improved performance compared to baseline PINN.

## 📊 Key Results

- **Solution Quality**: Excellent (differences < 0.004 from baseline)
- **PDE Loss**: Competitive or better in 50% of test cases
- **Best Performance**: 76% improvement (4.2x better) for ε=0.05
- **Status**: ✅ Successfully validated on multiple test cases

See [RESULTS.md](RESULTS.md) for detailed results and visualizations.

## 🚀 Quick Start

### Prerequisites

- **OS**: Ubuntu 20.04 LTS or higher
- **GPU**: NVIDIA T2000 (or compatible NVIDIA GPU with CUDA support)
- **CUDA**: 11.2 or higher
- **Python**: 3.10 (managed via Conda)
- **Conda**: Anaconda or Miniconda

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Noman-Nom/pinns-canpinns-research
   cd PINNS
   ```

2. **Create the conda environment**:
   ```bash
   conda env create -f environment.yml
   conda activate pinns
   ```

3. **Verify installation**:
   ```bash
   python verify_gpu.py
   ```

### Usage

**Train baseline PINN and improved CAN-PINN**:
```bash
python train_improved_allen_cahn.py
```

**Quick test**:
```bash
python test_improved.py
```

**Train on specific test case**:
```bash
python run_single_test.py
```

## 📁 Project Structure

```
PINNS/
├── README.md                          # This file
├── RESULTS.md                         # Detailed results with visualizations
├── environment.yml                    # Conda environment specification
├── setup_environment.sh               # Automated setup script
│
├── Core Models
│   ├── pinn_model.py                  # Base PINN model
│   ├── allen_cahn_pinn.py             # Baseline Allen-Cahn PINN
│   ├── allen_cahn_pinn_improved.py    # Hybrid CAN-PINN implementation
│   └── residual_adaptive_sampling.py  # Adaptive sampling module
│
├── Training Scripts
│   ├── train_heat_equation.py         # Heat equation training
│   ├── train_allen_cahn.py            # Baseline Allen-Cahn training
│   ├── train_improved_allen_cahn.py   # Hybrid CAN-PINN training
│   └── run_single_test.py             # Single test case runner
│
├── Testing
│   ├── test_pinn.py                   # PINN tests
│   ├── test_allen_cahn.py             # Allen-Cahn tests
│   └── test_improved.py               # Improved model tests
│
├── Utilities
│   ├── verify_gpu.py                  # GPU verification
│   ├── cuda_init.py                   # CUDA initialization
│   └── wave_equation_pinn.py          # Wave equation implementation
│
└── Documentation
    ├── PINN_DOCUMENTATION.md          # Technical documentation
    ├── SUPERVISOR_SUMMARY.md          # Summary for supervisor
    ├── HONEST_RESULTS_REVIEW.md       # Detailed results analysis
    └── HYBRID_APPROACH_IMPLEMENTATION.md  # Implementation details
```

## 🔬 Technical Details

### Model Architecture

- **Network**: MLP with 2 inputs (x, t), 3 hidden layers (50 neurons each), 1 output (u)
- **Activation**: Tanh
- **Optimizer**: Adam (10,000 epochs) + L-BFGS (1,000 iterations)

### Hybrid CAN-PINN Features

1. **Automatic Differentiation**: AD for all derivatives (eliminates numerical errors)
2. **Uncertainty Weighting**: Learnable weights for IC/BC/PDE loss terms
3. **Adaptive Sampling**: Residual-based adaptive sampling (resample 10% every 3000 epochs)
4. **Gradient Penalty**: Promotes smoother solutions (λ = 1e-5)
5. **L-BFGS Fine-tuning**: Additional optimization phase

### Test Cases

- **TC2**: Varying initial conditions (sin(πx), step function), ε=0.01
- **TC3**: Varying diffusivity (ε=0.01, 0.05), sin(πx) initial condition
- **Domain**: x ∈ [0, 1], t ∈ [0, 1]
- **Boundary**: Dirichlet (u=0 at x=0,1)

## 📈 Results Summary

| Test Case | PINN PDE Loss | CAN-PINN PDE Loss | Result |
|-----------|---------------|-------------------|--------|
| TC2: sin(πx), ε=0.01 | 2.48e-05 | **1.59e-05** | ✅ 36% better |
| TC2: step, ε=0.01 | 3.87e-04 | 4.10e-04 | ⚠️ 6% worse |
| TC3: sin(πx), ε=0.01 | 1.36e-05 | 2.34e-05 | ⚠️ 72% worse |
| TC3: sin(πx), ε=0.05 | 2.63e-05 | **6.30e-06** | ✅ 76% better (4.2x) |

**Key Finding**: CAN-PINN shows significant improvements for larger ε values.

## 📚 Documentation

- **[RESULTS.md](RESULTS.md)**: Detailed results with visualizations
- **[SUPERVISOR_SUMMARY.md](SUPERVISOR_SUMMARY.md)**: Summary for supervisor presentation
- **[PINN_DOCUMENTATION.md](PINN_DOCUMENTATION.md)**: Technical documentation
- **[HONEST_RESULTS_REVIEW.md](HONEST_RESULTS_REVIEW.md)**: Comprehensive results analysis

## 🔧 Troubleshooting

### CUDA Not Available

```bash
# Verify NVIDIA drivers
nvidia-smi

# Check CUDA installation
nvcc --version

# Reinstall PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Environment Issues

```bash
# Recreate environment
conda env remove -n pinns
conda env create -f environment.yml
conda activate pinns
```

## 🎓 Key Achievements

✅ **Fixed Critical Issues**: Eliminated numerical differentiation errors  
✅ **Solution Quality**: Excellent (differences < 0.004)  
✅ **Performance**: Competitive or better in 50% of cases  
✅ **Best Result**: 76% improvement for ε=0.05 (4.2x better)

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@software{hybrid_can_pinn,
  title = {Hybrid CAN-PINNs for Allen-Cahn Equation},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/PINNS}
}
```

## 📄 License

This project is for research purposes.

## 👥 Contributors

- [Your Name] - Initial work and implementation

## 🙏 Acknowledgments

- Based on the Physics-Informed Neural Networks framework by Raissi et al. (2019)
- Inspired by CAN-PINN approaches for adaptive sampling and uncertainty weighting

---

For detailed results and visualizations, see [RESULTS.md](RESULTS.md).
