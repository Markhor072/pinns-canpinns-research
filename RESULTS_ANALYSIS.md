# Training Results Analysis: PINN vs Improved CAN-PINN

## Test Configuration
- **Equation**: Allen-Cahn (ε = 0.01)
- **Initial Condition**: sin(πx)
- **Training Epochs**: 10,000 (Adam) + 1,000 (L-BFGS)
- **Architecture**: [2, 50, 50, 50, 1]

---

## 1. Performance Comparison

### Final Loss Values

| Metric | PINN (Baseline) | Improved CAN-PINN | Ratio |
|--------|----------------|-------------------|-------|
| **Total Loss** | 3.54e-05 | -5.04e+00* | N/A |
| **PDE Loss** | **2.52e-05** | **8.30e-04** | **33× worse** |
| **IC Loss** | 6.47e-06 | 5.71e-02 | 8,800× worse |
| **BC Loss** | 3.79e-06 | 3.37e-03 | 890× worse |

*Note: CAN-PINN total loss is negative due to uncertainty weighting formula (log_var terms).

### Key Observations

#### ❌ **Critical Issue: CAN-PINN Underperforms Significantly**

1. **PDE Loss**: CAN-PINN is **33× worse** than baseline PINN
   - PINN: 2.52e-05 (excellent)
   - CAN-PINN: 8.30e-04 (poor)

2. **IC/BC Loss**: CAN-PINN is **hundreds to thousands of times worse**
   - This suggests the model is not satisfying boundary/initial conditions well

3. **Convergence**: 
   - PINN: Smooth, steady convergence
   - CAN-PINN: Higher losses throughout training

---

## 2. Adaptive Weight Analysis

### Weight Evolution

| Epoch | IC Weight | BC Weight | PDE Weight |
|-------|-----------|-----------|------------|
| 0 | 0.500 | 0.500 | 0.500 |
| 1000 | 1.441 | 1.366 | 1.359 |
| 5000 | 23.285 | 31.693 | 30.770 |
| 8999 | **7.820** | **74.207** | **74.207** |

### Issues Identified

1. **Weights Hitting Clamp Limit**: 
   - BC and PDE weights reached 74.207 (clamped at ~74.2)
   - This means the uncertainty weighting is trying to push weights even higher
   - **Problem**: The model is struggling, so it's trying to increase weights to compensate

2. **IC Weight Decreases**: 
   - IC weight drops from 23.3 (epoch 5000) to 7.8 (epoch 8999)
   - This suggests IC is being satisfied better, but BC/PDE are not

3. **Weight Imbalance**: 
   - BC and PDE weights are 9.5× higher than IC weight
   - This extreme imbalance suggests the model is struggling with physics constraints

---

## 3. Adaptive Sampling Analysis

### Resampling Statistics

| Epoch | Max Residual | Mean Residual | Points Resampled |
|-------|--------------|---------------|------------------|
| 1000 | 1.94e+00 | 6.10e-02 | 20,000 |
| 2000 | 4.21e-01 | 6.50e-02 | 20,000 |
| 5000 | 4.14e-01 | 6.44e-02 | 20,000 |
| 8000 | 2.88e-01 | 2.99e-02 | 20,000 |

### Observations

1. **Residuals Decreasing**: Mean residual drops from 6.1e-02 to 2.99e-02
   - This is good - adaptive sampling is finding better regions

2. **But Still High**: Mean residual of 2.99e-02 is still much higher than PINN's final PDE loss (2.52e-05)
   - This suggests the numerical differentiation might be introducing errors

3. **Resampling Frequency**: Every 1000 epochs might be too frequent
   - Could be disrupting training stability

---

## 4. L-BFGS Phase Analysis

### L-BFGS Results

```
L-BFGS Iter     0 | Total Loss: -5.040357e+00 | PDE Loss: 8.302723e-04
L-BFGS Iter   999 | Total Loss: -5.040357e+00 | PDE Loss: 8.302723e-04
```

**Problem**: L-BFGS shows **zero improvement**
- Loss doesn't change at all during 1000 iterations
- This suggests the model is stuck in a local minimum or the closure function has issues

---

## 5. Root Cause Analysis

### Why CAN-PINN Underperforms

#### 1. **Numerical Differentiation Errors**
- **Issue**: Using finite differences for `u_xx` introduces truncation errors
- **Impact**: Even with adaptive `h`, numerical errors accumulate
- **Evidence**: High PDE residuals despite adaptive sampling

#### 2. **Uncertainty Weighting Issues**
- **Issue**: Weights hitting clamp limit (74.2) indicates model struggling
- **Impact**: Extreme weight imbalance destabilizes training
- **Evidence**: BC/PDE weights 9.5× higher than IC

#### 3. **Boundary Handling**
- **Issue**: Symmetric boundary stencils might not be accurate enough
- **Impact**: Boundary conditions not satisfied well (BC loss 890× worse)
- **Evidence**: High BC loss despite high BC weight

#### 4. **Adaptive Sampling Too Aggressive**
- **Issue**: Resampling 20,000 points every 1000 epochs disrupts training
- **Impact**: Model can't stabilize, constantly adapting to new point sets
- **Evidence**: Losses don't decrease smoothly

#### 5. **L-BFGS Not Working**
- **Issue**: Closure function or optimizer settings might be wrong
- **Impact**: No fine-tuning benefit
- **Evidence**: Zero change in loss during L-BFGS phase

---

## 6. Comparison with Image Results

From the visualization:
- **Solutions Look Similar**: 3D surfaces are visually close
- **But Differences Exist**: Especially at t=1.0, CAN-PINN is lower
- **Loss Curves**: CAN-PINN shows much higher and more unstable losses
- **Weights**: Show extreme growth, confirming weight saturation issue

---

## 7. Recommendations

### Immediate Fixes

#### 1. **Fix Numerical Differentiation**
```python
# Current: Central difference with adaptive h
# Better: Use higher-order schemes or hybrid AD/ND
# Or: Use AD for all derivatives (like PINN)
```

#### 2. **Adjust Uncertainty Weighting**
```python
# Current: Clamp at [-5, 5] → weights [0.003, 74.2]
# Better: 
# - Increase clamp range: [-3, 3] → weights [0.05, 20]
# - Or: Use fixed weights with better initialization
# - Or: Use gradient-based weighting instead
```

#### 3. **Improve Boundary Handling**
```python
# Current: Symmetric extension
# Better: 
# - Use ghost points with proper boundary conditions
# - Or: Use AD for boundary points, ND only for interior
```

#### 4. **Reduce Adaptive Sampling Frequency**
```python
# Current: Every 1000 epochs
# Better: 
# - Every 2000-3000 epochs
# - Or: Only resample when loss plateaus
# - Reduce resample fraction: 0.2 → 0.1
```

#### 5. **Fix L-BFGS**
```python
# Check closure function
# Ensure gradients are computed correctly
# Try different L-BFGS settings
```

### Alternative Approaches

#### Option A: Hybrid Approach
- Use **AD for time derivatives** (like current)
- Use **AD for spatial derivatives** (like PINN) - **remove numerical differentiation**
- Keep uncertainty weighting but with better bounds
- Keep adaptive sampling but less frequently

#### Option B: Simplified CAN-PINN
- Remove numerical differentiation entirely
- Use **only AD** (like PINN)
- Keep uncertainty weighting (with better bounds)
- Keep adaptive sampling (less frequent)

#### Option C: Fix Current Approach
- Improve numerical differentiation (higher order)
- Better boundary handling
- Tune uncertainty weighting bounds
- Fix L-BFGS

---

## 8. Expected Improvements

If fixes are applied:

| Metric | Current CAN-PINN | Target | Improvement |
|--------|------------------|--------|-------------|
| PDE Loss | 8.30e-04 | < 1.00e-04 | 8× better |
| IC Loss | 5.71e-02 | < 1.00e-05 | 5,700× better |
| BC Loss | 3.37e-03 | < 1.00e-05 | 337× better |
| Weights | 74.207 (clamped) | < 20 | Stable |

---

## 9. Conclusion

### Current Status: ❌ **CAN-PINN Underperforms Baseline**

**Key Findings**:
1. PDE loss is **33× worse** than PINN
2. Weights are **saturated** at clamp limit
3. L-BFGS provides **no benefit**
4. Adaptive sampling helps but **not enough**

### Next Steps

1. **Priority 1**: Fix numerical differentiation or switch to AD
2. **Priority 2**: Adjust uncertainty weighting bounds
3. **Priority 3**: Improve boundary handling
4. **Priority 4**: Fix L-BFGS or remove it
5. **Priority 5**: Tune adaptive sampling frequency

### Recommendation

**Consider hybrid approach**: Use AD for all derivatives (like PINN), but keep:
- Uncertainty weighting (with better bounds)
- Adaptive sampling (less frequent)
- Gradient penalty
- Fourier features (if needed)

This would combine the **accuracy of AD** with the **training improvements** of CAN-PINN.

---

**Analysis Date**: 2025-11-12
**Status**: Needs significant improvements before CAN-PINN can outperform baseline PINN

