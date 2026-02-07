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

    def disabled_compile(model=None, *args, **kwargs):
        """
        No-op compile that just returns the model/function unchanged.
        Works as both a function and a decorator.
        """
        def wrapper(func_or_model):
            print(f"torch.compile called but disabled on Windows - returning {type(func_or_model).__name__} unchanged")
            return func_or_model

        # If called as decorator: @torch.compile(...)
        if model is None:
            return wrapper

        # If called as function: torch.compile(model, ...)
        print(f"torch.compile called but disabled on Windows - returning {type(model).__name__} unchanged")
        return model

    torch.compile = disabled_compile

    # Disable torch._dynamo
    try:
        torch._dynamo.config.suppress_errors = True  # type: ignore
        torch._dynamo.config.disable = True  # type: ignore
        print("torch._dynamo disabled successfully")
    except (AttributeError, ImportError):
        print("Could not disable torch._dynamo")

    print("Windows Triton workaround active - torch.compile disabled")
