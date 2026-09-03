"""xformers on Blackwell (RTX 50xx, sm_120): the wheels' flash-attention ops
(flash3 = hopper kernels, flash = sm80 kernels) are picked by the dispatcher
for cc >= 9.0 and then fail on sm_120 ("no kernel image" / "invalid argument").
The CUTLASS memory-efficient kernel in the same wheel runs fine there and
matches SDPA to fp16 precision, so this shim tells the dispatcher the flash
ops are unsupported. Import it before anything that imports xformers/TRELLIS.
No-op on older GPUs."""
import torch

if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10:
    try:
        from xformers.ops import fmha
        for name in ("flash3", "flash"):
            mod = getattr(fmha, name, None)
            for cls_name in ("FwOp", "BwOp"):
                cls = getattr(mod, cls_name, None) if mod is not None else None
                if cls is not None:
                    cls.not_supported_reasons = classmethod(lambda c, d: ["disabled on sm_120 (blackwell_shim)"])
        print("[blackwell_shim] xformers flash ops disabled -> CUTLASS attention", flush=True)
    except Exception as e:                       # xformers absent -> nothing to do
        print("[blackwell_shim] skipped:", repr(e), flush=True)
