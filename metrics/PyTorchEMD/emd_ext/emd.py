import torch
import os
from torch.autograd import Function

# Load the compiled C++ and CUDA extension
# This assumes the compiled library is named 'emd_ext_cuda' and is in the correct path
try:
    # Use torch.utils.cpp_extension.load or importlib if outside a typical setup environment
    # For a setup.py build, it will usually be placed under the package path.
    # We will assume it gets placed in the same path as the wrapper for simplicity for now.
    
    # We load the library compiled by setup.py. The actual name might be something like 
    # emd_ext_cuda.cpython-312-x86_64-linux-gnu.so
    # When installed, it can be imported directly. We use a dynamic import approach for robustness.
    
    # First, check if we are already installed/imported
    # Try importing the compiled module directly from the source directory, this usually works post-install
    
    # A cleaner approach is to use the actual package name created by setup.py:
    import emd_ext_cuda as emd_cuda
    print("Successfully imported emd_ext_cuda module.")
except ImportError:
    # If the import fails, print an informative message
    print("Warning: Could not import the compiled PyTorch CUDA extension 'emd_ext_cuda'.")
    print("Please ensure you have run 'python setup.py install' in the directory containing setup.py.")
    emd_cuda = None
    
# Check if the module loaded successfully before defining the Functions
if emd_cuda is not None:
    
    class ApproxMatch(Function):
        """
        Approximate Earth Mover's Distance Match (Forward Pass)
        Input: 
            xyz1 (B, N1, 3) 
            xyz2 (B, N2, 3) 
        Output: 
            match (B, N2, N1) - The approximate transportation plan
        """
        @staticmethod
        def forward(ctx, xyz1, xyz2):
            match = emd_cuda.approxmatch_forward(xyz1, xyz2)
            ctx.save_for_backward(xyz1, xyz2, match)
            return match

        @staticmethod
        def backward(ctx, grad_match):
            # No backward for ApproxMatch itself, it's typically treated as a constant flow.
            # The MatchCost backward uses the match output, but ApproxMatch itself doesn't need
            # gradients through it to matchcost's inputs (xyz1, xyz2) in the standard use case.
            # Return None for the two inputs (xyz1, xyz2)
            return None, None 

    
    class MatchCost(Function):
        """
        Earth Mover's Distance Match Cost (Forward and Backward Pass)
        Input: 
            xyz1 (B, N1, 3) 
            xyz2 (B, N2, 3) 
            match (B, N2, N1) - The transportation plan from ApproxMatch
        Output: 
            cost (B)
        """
        @staticmethod
        def forward(ctx, xyz1, xyz2, match):
            cost = emd_cuda.matchcost_forward(xyz1, xyz2, match)
            ctx.save_for_backward(xyz1, xyz2, match)
            return cost

        @staticmethod
        def backward(ctx, grad_cost):
            xyz1, xyz2, match = ctx.saved_tensors
            
            grad1, grad2 = emd_cuda.matchcost_backward(grad_cost.contiguous(), xyz1, xyz2, match)
            
            # MatchCost is NOT differentiable w.r.t 'match' in this original implementation
            # It returns gradients for xyz1 and xyz2 only.
            # The return order must match the forward input order: grad_xyz1, grad_xyz2, grad_match
            return grad1, grad2, None 

    # Public functions to use in your script
    def approx_match(xyz1, xyz2):
        return ApproxMatch.apply(xyz1, xyz2)
        
    def match_cost(xyz1, xyz2, match):
        return MatchCost.apply(xyz1, xyz2, match)

    def emd_loss(xyz1, xyz2):
        # A utility function to combine them
        match = approx_match(xyz1, xyz2)
        cost = match_cost(xyz1, xyz2, match)
        return cost
        
else:
    # Fallback/Dummy functions if the CUDA module didn't load
    def approx_match(*args):
        raise RuntimeError("CUDA extension not loaded. Cannot run approx_match.")
    def match_cost(*args):
        raise RuntimeError("CUDA extension not loaded. Cannot run match_cost.")
    def emd_loss(*args):
        raise RuntimeError("CUDA extension not loaded. Cannot run emd_loss.")


# Export the primary functions
__all__ = ['approx_match', 'match_cost', 'emd_loss']
