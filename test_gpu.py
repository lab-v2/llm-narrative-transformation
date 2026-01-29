#!/usr/bin/env python3
import os


def _format_gb(num_bytes):
    return num_bytes / (1024 ** 3)


def main():
    # Environment
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    print(f"UDA_VISIBLE_DEVICES: {cuda_visible}")
    print()

    # PyTorch Information
    try:
        import torch  # type: ignore
    except Exception as exc:
        print("PyTorch Information:")
        print(f"  - PyTorch import failed: {type(exc).__name__}: {exc}")
        return

    try:
        cudnn_version = torch.backends.cudnn.version()
    except Exception:
        cudnn_version = None

    print("PyTorch Information:")
    print(f"  - PyTorch version: {torch.__version__}")
    print(f"  - CUDA compiled version: {torch.version.cuda}")
    print(f"  - cuDNN version: {cudnn_version}")
    print()

    # Device Detection
    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_available else 0
    print("Device Detection:")
    print(f"  - CUDA available: {cuda_available}")
    print(f"  - CUDA device count: {device_count}")

    if cuda_available and device_count > 0:
        idx = 0
        name = torch.cuda.get_device_name(idx)
        props = torch.cuda.get_device_properties(idx)
        total_gb = _format_gb(props.total_memory)
        print(f"  - Device {idx}: {name}")
        print(f"    Memory: {total_gb:.1f} GB")
        print(f"    SMs: {props.multi_processor_count}")
        print(f"    Compute Capability: {props.major}.{props.minor}")
    print()

    # Testing GPU Operations
    print("Testing GPU Operations:")
    if not cuda_available:
        print("  - CUDA not available; skipping test")
        return

    try:
        device = torch.device("cuda:0")
        a = torch.randn(1000, 1000, device=device)
        b = torch.randn(1000, 1000, device=device)
        c = a @ b
        torch.cuda.synchronize()
        print(f"  - Matrix multiplication successful: {c.shape}")
        allocated_gb = _format_gb(torch.cuda.memory_allocated(device))
        cached_gb = _format_gb(torch.cuda.memory_reserved(device))
        print(f"  - GPU memory allocated: {allocated_gb:.2f} GB")
        print(f"  - GPU memory cached: {cached_gb:.2f} GB")
        print("✓ GPU test passed!")
    except Exception as exc:
        print(f"  - GPU test failed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
