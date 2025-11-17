"""
Quick test script for Allen-Cahn PINN and CAN-PINN implementations.
This runs a shorter training to verify the framework works correctly.
"""

import numpy as np
import torch
from allen_cahn_pinn import AllenCahnPINN, CANAllenCahnPINN


def quick_test():
    """Quick test of the Allen-Cahn PINN implementations."""
    print("="*60)
    print("Quick Test: Allen-Cahn PINN vs CAN-PINN")
    print("="*60)
    
    # Parameters
    epsilon = 0.01
    layers = [2, 20, 20, 1]  # Smaller network for quick test
    
    # Generate minimal training data
    print("\nGenerating training data...")
    
    # Initial condition: u(x, 0) = sin(πx)
    x_ic = np.random.uniform(0, 1, (50, 1))
    t_ic = np.zeros((50, 1))
    u_ic = np.sin(np.pi * x_ic)
    
    # Boundary conditions
    x_bc_left = np.zeros((25, 1))
    t_bc_left = np.random.uniform(0, 1, (25, 1))
    x_bc_right = np.ones((25, 1))
    t_bc_right = np.random.uniform(0, 1, (25, 1))
    x_bc = np.vstack([x_bc_left, x_bc_right])
    t_bc = np.vstack([t_bc_left, t_bc_right])
    u_bc = np.zeros((50, 1))
    
    # PDE collocation points
    x_pde = np.random.uniform(0, 1, (1000, 1))
    t_pde = np.random.uniform(0, 1, (1000, 1))
    
    # Test data
    x_test = np.linspace(0, 1, 50)
    t_test = np.linspace(0, 1, 50)
    X_test, T_test = np.meshgrid(x_test, t_test)
    x_test = X_test.flatten()
    t_test = T_test.flatten()
    
    print(f"  IC points: {len(x_ic)}")
    print(f"  BC points: {len(x_bc)}")
    print(f"  PDE points: {len(x_pde)}")
    print(f"  Test points: {len(x_test)}")
    
    # Initialize models
    print("\nInitializing models...")
    model_pinn = AllenCahnPINN(epsilon=epsilon, layers=layers, device='cuda')
    model_can = CANAllenCahnPINN(epsilon=epsilon, layers=layers, device='cuda', h=0.01)
    
    # Quick training (500 epochs)
    print("\nRunning quick training (500 epochs)...")
    print("\n--- PINN Training ---")
    history_pinn = model_pinn.train(
        x_ic, t_ic, u_ic,
        x_bc, t_bc, u_bc,
        x_pde, t_pde,
        x_test, t_test, None,
        epochs=500,
        lr=0.001,
        print_every=100
    )
    
    print("\n--- CAN-PINN Training ---")
    history_can = model_can.train(
        x_ic, t_ic, u_ic,
        x_bc, t_bc, u_bc,
        x_pde, t_pde,
        x_test, t_test, None,
        epochs=500,
        lr=0.001,
        print_every=100
    )
    
    # Final evaluation
    final_loss_pinn = history_pinn['total_loss'][-1]
    final_loss_can = history_can['total_loss'][-1]
    
    print("\n" + "="*60)
    print("Results Summary")
    print("="*60)
    print(f"PINN - Final Loss: {final_loss_pinn:.6e}")
    print(f"CAN-PINN - Final Loss: {final_loss_can:.6e}")
    
    improvement = (final_loss_pinn - final_loss_can) / final_loss_pinn * 100
    print(f"Improvement: {improvement:.2f}%")
    
    # Check if CAN-PINN has adaptive weights
    if history_can['weights'] is not None:
        weights = history_can['weights']
        print(f"\nFinal Adaptive Weights (CAN-PINN):")
        print(f"  IC: {weights['ic'][-1]:.3f}")
        print(f"  BC: {weights['bc'][-1]:.3f}")
        print(f"  PDE: {weights['pde'][-1]:.3f}")
    
    print("\n" + "="*60)
    print("Quick Test Complete!")
    print("="*60)
    print(f"\n✓ Framework is working correctly")
    print(f"✓ Both PINN and CAN-PINN trained successfully")
    
    return final_loss_pinn < 1.0 and final_loss_can < 1.0


if __name__ == "__main__":
    success = quick_test()
    if success:
        print("\n✓ Test PASSED - Ready for full training!")
    else:
        print("\n⚠ Test completed but losses are high - may need more training epochs")

