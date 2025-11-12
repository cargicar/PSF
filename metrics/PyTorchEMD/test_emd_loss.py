import torch
import os
import random

# Assuming the setup process installed the module successfully
# The setup.py defined the package 'emd_ext', which contains the wrapper 'emd.py'
try:
    from emd_ext.emd import approx_match, match_cost, emd_loss
except ImportError as e:
    print(f"Error importing EMD functions: {e}")
    print("Please ensure you have run 'python setup.py install' and the module 'emd_ext_cuda' compiled successfully.")
    exit()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == 'cpu':
    print("Warning: CUDA not available. Cannot test the CUDA extension.")
    exit()

def run_test():
    print(f"Running EMD test on device: {device}")
    
    # Configuration
    B = 2  # Batch size
    N1 = 1024 # Number of points in set 1
    N2 = 2048 # Number of points in set 2
    
    # 1. Create random input tensors (must be float and on CUDA)
    # xyz1: (B, N1, 3) - differentiable input
    xyz1 = torch.rand(B, N1, 3, dtype=torch.float32, device=device, requires_grad=True) 
    # xyz2: (B, N2, 3) - differentiable input
    xyz2 = torch.rand(B, N2, 3, dtype=torch.float32, device=device, requires_grad=True) 

    # 2. ApproxMatch Forward Pass
    print(f"Input shapes: xyz1={xyz1.shape}, xyz2={xyz2.shape}")
    
    try:
        # Run ApproxMatch
        match = approx_match(xyz1, xyz2)
        print(f"ApproxMatch output shape (match): {match.shape}")
        
        # Expected shape is (B, N2, N1) -> (2, 2048, 1024)
        assert match.shape == (B, N2, N1)
        print("ApproxMatch test successful.")

        # 3. MatchCost Forward Pass
        cost = match_cost(xyz1, xyz2, match)
        print(f"MatchCost output shape (cost): {cost.shape}")
        
        # Expected shape is (B) -> (2,)
        assert cost.shape == (B,)
        print("MatchCost forward test successful.")
        
        # 4. Backward Pass (Compute gradients)
        
        # Create a dummy gradient for the cost (e.g., all ones)
        grad_output = torch.ones_like(cost)
        
        # Compute gradients (the backward pass is implicitly called)
        cost.backward(grad_output)
        
        # Check if gradients were computed for xyz1 and xyz2
        if xyz1.grad is not None and xyz2.grad is not None:
            print("Backward pass successful. Gradients computed.")
            print(f"Grad(xyz1) shape: {xyz1.grad.shape}")
            print(f"Grad(xyz2) shape: {xyz2.grad.shape}")
            assert xyz1.grad.shape == xyz1.shape
            assert xyz2.grad.shape == xyz2.shape
        else:
            print("Backward pass FAILED: Gradients were not computed for inputs.")
            
        print("\nFull EMD pipeline test finished.")

    except RuntimeError as e:
        print(f"\nRUNTIME ERROR during test: {e}")
        print("This usually means the CUDA kernels failed or the module was not correctly linked.")


if __name__ == '__main__':
    # Ensure all files are set up correctly before running the test script
    # This test script assumes you are running it from a separate directory 
    # where the 'emd_ext' package is importable (i.e., after running setup.py install/develop).
    run_test()