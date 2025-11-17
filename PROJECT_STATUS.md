# Project Status: CAN-PINN for Allen-Cahn Equation

## Current Status: ✅ Improvements Implemented and Ready for Experiments

## Project Overview

**Goal**: Develop and improve CAN-PINN (Conservative Allen-Cahn Neural Physics-Informed Neural Network) to solve the Allen-Cahn equation, comparing with standard PINN approaches.

## Completed Work

### Phase 1: Environment Setup ✅
- ✅ Conda environment with PyTorch and TensorFlow
- ✅ GPU verification (NVIDIA T2000)
- ✅ All dependencies installed
- ✅ Environment tested and verified

### Phase 2: Basic PINN Implementation ✅
- ✅ Heat Equation PINN (Raissi et al. 2019)
- ✅ Wave Equation PINN
- ✅ Automatic differentiation
- ✅ Loss function (IC, BC, PDE)
- ✅ L2 error tracking
- ✅ Validation against analytical solutions

**Results**:
- Heat Equation: L2 error 0.024 (best: 1.8e-04)
- Framework validated and working

### Phase 3: Allen-Cahn PINN Implementation ✅
- ✅ Standard PINN for Allen-Cahn
- ✅ Original CAN-PINN implementation
- ✅ Numerical differentiation for spatial derivatives
- ✅ Adaptive loss weighting (gradient-based)
- ✅ Multiple test cases (varying IC, ε, domains)

**Results**:
- Standard PINN: Excellent (loss: 3.8e-05 to 5.4e-05)
- Original CAN-PINN: Underperformed (loss: 7.7e-02 to 8.3e-02)
- Identified issues: weight saturation, poor boundary handling, fixed h

### Phase 4: Improved CAN-PINN Implementation ✅
- ✅ Adaptive step size (h) based on sampling density
- ✅ Improved boundary handling (symmetric extension)
- ✅ Uncertainty-based loss weighting (learned weights)
- ✅ Gradient penalty regularization
- ✅ L-BFGS fine-tuning option
- ✅ Residual-based adaptive sampling (RBAS)
- ✅ Fourier feature encoding (optional)
- ✅ Comprehensive training scripts
- ✅ Quick test verified working

## File Structure

```
PINNS/
├── Core Implementation
│   ├── pinn_model.py                    # Base PINN class
│   ├── allen_cahn_pinn.py               # Standard PINN & CAN-PINN
│   ├── allen_cahn_pinn_improved.py      # Improved CAN-PINN ⭐
│   ├── residual_adaptive_sampling.py    # Adaptive sampling ⭐
│   └── wave_equation_pinn.py            # Wave equation PINN
│
├── Training Scripts
│   ├── train_heat_equation.py           # Heat equation training
│   ├── train_allen_cahn.py              # Original CAN-PINN training
│   ├── train_improved_allen_cahn.py     # Improved CAN-PINN training ⭐
│   ├── run_single_test.py               # Single test case runner
│   ├── test_pinn.py                     # Quick PINN test
│   ├── test_allen_cahn.py               # Quick Allen-Cahn test
│   └── test_improved.py                 # Quick improved test ⭐
│
├── Documentation
│   ├── README.md                        # Project overview
│   ├── PINN_DOCUMENTATION.md            # PINN implementation docs
│   ├── TASK1_SUMMARY.md                 # Task 1 results
│   ├── TASK2_SUMMARY.md                 # Task 2 results
│   ├── IMPROVEMENTS_GUIDE.md            # Improvements documentation ⭐
│   ├── SUPERVISOR_PRESENTATION.md       # Meeting guide ⭐
│   └── PROJECT_STATUS.md                # This file ⭐
│
├── Results (Previous)
│   ├── heat_equation_results.png        # Heat equation results
│   ├── training_history.png             # Training history
│   ├── allen_cahn_tc2_eps0.01_icsin.png
│   ├── allen_cahn_tc2_eps0.01_icstep.png
│   ├── allen_cahn_tc3_eps0.01_icsin.png
│   ├── allen_cahn_tc3_eps0.05_icsin.png
│   └── allen_cahn_tc4_eps0.01_icsin.png
│
└── Setup
    ├── environment.yml                  # Conda environment
    ├── setup_environment.sh             # Setup script
    └── verify_gpu.py                    # GPU verification
```

⭐ = New/Improved files

## Key Improvements Implemented

### 1. Adaptive Step Size
- Computes `h` from data sampling density
- Adapts to different domains and sampling strategies
- More accurate numerical differentiation

### 2. Improved Boundary Handling
- Symmetric extension instead of clamping
- Preserves gradient flow
- Reduces numerical bias

### 3. Uncertainty-Based Loss Weighting
- Learnable uncertainty weights
- Prevents weight saturation
- Natural adaptation during training

### 4. Gradient Penalty
- Promotes smoother solutions
- Reduces oscillations at interfaces
- Helps with sharp interfaces

### 5. L-BFGS Fine-Tuning
- Two-phase training (Adam → L-BFGS)
- Better convergence for physics constraints
- Finer optimization

### 6. Residual-Based Adaptive Sampling
- Focuses on high-residual regions
- Better efficiency
- Improved accuracy

### 7. Fourier Features (Optional)
- Better high-frequency capture
- Useful for sharp interfaces
- Configurable frequency range

## Next Steps

### Immediate (This Week)
1. ✅ Present improvements to supervisor
2. ⏳ Get feedback on approach
3. ⏳ Run full experiments (10,000 epochs)
4. ⏳ Initial results analysis

### Short-term (Next 2 Weeks)
1. ⏳ Compare improved CAN-PINN vs standard PINN
2. ⏳ Hyperparameter tuning
3. ⏳ Performance analysis
4. ⏳ Document results

### Long-term (Next Month)
1. ⏳ Paper preparation
2. ⏳ Additional experiments
3. ⏳ Publication submission

## Usage

### Quick Test
```bash
conda activate pinns
python test_improved.py
```

### Single Test Case
```bash
python run_single_test.py --test_case 2 --epsilon 0.01 --ic_type sin --epochs 10000
```

### Full Comparison
```bash
python train_improved_allen_cahn.py
```

## Expected Results

### Performance Targets
- Improved CAN-PINN loss < 2× PINN loss
- Stable training (no weight saturation)
- Better handling of sharp interfaces
- Faster or comparable training time

### Success Metrics
- Convergence: Smooth loss decrease
- Accuracy: Within 2-3× of PINN
- Stability: No weight saturation
- Robustness: Works on all test cases

## Documentation

### For Supervisor Meeting
- **SUPERVISOR_PRESENTATION.md**: Complete presentation guide
- **IMPROVEMENTS_GUIDE.md**: Technical implementation details
- **PROJECT_STATUS.md**: This file (overview)

### For Development
- **PINN_DOCUMENTATION.md**: PINN implementation details
- **TASK1_SUMMARY.md**: Task 1 results and analysis
- **TASK2_SUMMARY.md**: Task 2 results and analysis

## Contact and Support

### Questions?
- Check documentation files
- Review code comments
- Run test scripts
- Check previous results

### Issues?
- Verify GPU setup: `python verify_gpu.py`
- Check environment: `conda activate pinns`
- Test imports: `python test_improved.py`

---

**Last Updated**: Current
**Status**: ✅ Ready for experiments
**Next Action**: Run full training and compare results

