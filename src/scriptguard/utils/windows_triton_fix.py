"""
Windows Triton workaround - disable torch.compile globally
Import this module BEFORE any Unsloth/transformers imports
"""
import os
import platform

if platform.system() == "Windows":
    # Set environment variables
    os.environ["TORCH_COMPILE_DISABLE"] = "1"
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    # Monkey-patch torch.compile to do nothing
    import torch

    original_compile = torch.compile

    def disabled_compile(*args, **kwargs):
        """
        No-op compile that just returns the model/function unchanged.
        Works as both a function and a decorator.
        """
        # If called with a function as first arg, return it unchanged
        if args and callable(args[0]):
            print(f"torch.compile called but disabled on Windows - returning {type(args[0]).__name__} unchanged")
            return args[0]

        # Otherwise return a decorator that returns functions unchanged
        def decorator(func):
            print(f"torch.compile decorator called but disabled on Windows - returning {type(func).__name__} unchanged")
            return func
        return decorator

    torch.compile = disabled_compile

    # Disable torch._dynamo
    try:
        torch._dynamo.config.suppress_errors = True  # type: ignore
        torch._dynamo.config.disable = True  # type: ignore
        print("torch._dynamo disabled successfully")
    except (AttributeError, ImportError):
        print("Could not disable torch._dynamo")

    print("Windows Triton workaround active - torch.compile disabled")
