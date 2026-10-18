"""What the training container can actually do, measured rather than assumed.

    python3 /env/probe.py                                # the short gate, about a minute
    python3 /env/probe.py --checkpoint Qwen/Qwen3-1.7B   # adds one load and one forward
    python3 /env/probe.py --require-sm121                # fail unless sm_121 is compiled in

Why the last flag is opt-in: on this image the compiled architecture list stops at compute_120
while the device reports compute 12.1, and every kernel still executes because the driver
JIT-compiles the embedded PTX. That is a usable machine, so failing on the missing entry would
be a false alarm. On a machine whose driver cannot JIT, the kernels fail at the launch, and the
checks below catch that directly - which is why the expensive part of this probe is a matmul and
a backward pass, not a string comparison against `torch.cuda.get_arch_list()`.

Exit status is 0 only if every requested check passed, so this is usable as a gate step.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Callable


def _timed(name: str, fn: Callable[[], str]) -> bool:
    started = time.time()
    try:
        detail = fn()
    except Exception as exc:  # noqa: BLE001 - the point is to report, not to classify
        print(f"FAIL  {name:22s} {time.time() - started:6.2f}s  {type(exc).__name__}: {str(exc)[:200]}")
        return False
    print(f"PASS  {name:22s} {time.time() - started:6.2f}s  {detail}")
    return True


def check_device() -> str:
    import torch

    props = torch.cuda.get_device_properties(0)
    archs = torch.cuda.get_arch_list()
    return (
        f"{props.name} | compute {props.major}.{props.minor} | "
        f"{props.total_memory / 2**30:.1f} GiB | archs={archs} | "
        f"sm_121_compiled={'sm_121' in archs}"
    )


def check_fp32_exactness() -> str:
    """A float32 product is exact to float32 only when TF32 is off.

    The NGC images export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1, so the default "fp32" matmul is
    TF32 and carries about 1e-4 relative error. Reporting that as a hardware fault is the
    mistake this check prevents: with TF32 disabled the same product is exact to about 1e-7.
    """
    import torch

    torch.manual_seed(0)
    a = torch.randn(1024, 1024, device="cuda")
    b = torch.randn(1024, 1024, device="cuda")
    reference = a.cpu() @ b.cpu()
    default_tf32 = torch.backends.cuda.matmul.allow_tf32
    err_tf32 = ((a @ b).cpu() - reference).abs().max().item() / reference.abs().max().item()
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        err_ieee = ((a @ b).cpu() - reference).abs().max().item() / reference.abs().max().item()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = default_tf32
    if err_ieee > 1e-6:
        raise AssertionError(f"float32 matmul is not exact with TF32 off: {err_ieee:.2e}")
    return f"tf32_default={default_tf32} err_tf32={err_tf32:.2e} err_ieee={err_ieee:.2e}"


def check_bf16_matmul() -> str:
    import torch

    torch.manual_seed(0)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
        out = a @ b
    torch.cuda.synchronize()
    return f"4096^3 -> {tuple(out.shape)} {out.dtype}"


def check_throughput() -> str:
    import torch

    a = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
    torch.cuda.synchronize()
    started = time.time()
    for _ in range(50):
        a @ b
    torch.cuda.synchronize()
    seconds = time.time() - started
    tflops = 50 * 2 * 2048**3 / seconds / 1e12
    return f"{tflops:.1f} TFLOP/s (bf16, 2048^3 x50)"


def check_sdpa() -> str:
    import torch

    q = torch.randn(2, 8, 512, 64, device="cuda", dtype=torch.bfloat16)
    out = torch.nn.functional.scaled_dot_product_attention(q, q, q, is_causal=True)
    torch.cuda.synchronize()
    if not torch.isfinite(out).all():
        raise AssertionError("attention produced non-finite values")
    return f"out={tuple(out.shape)} finite=True"


def check_backward() -> str:
    import torch

    net = (
        torch.nn.Sequential(torch.nn.Linear(512, 1024), torch.nn.GELU(), torch.nn.Linear(1024, 512))
        .cuda()
        .to(torch.bfloat16)
    )
    optimiser = torch.optim.AdamW(net.parameters(), lr=1e-3)
    x = torch.randn(32, 512, device="cuda", dtype=torch.bfloat16)
    losses = []
    for _ in range(10):
        loss = net(x).float().pow(2).mean()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        losses.append(loss.item())
    torch.cuda.synchronize()
    finite = all(
        p.grad is not None and torch.isfinite(p.grad.float()).all().item() for p in net.parameters()
    )
    if not finite:
        raise AssertionError("a gradient was missing or non-finite")
    if losses[-1] >= losses[0]:
        raise AssertionError(f"loss did not fall: {losses[0]:.4f} -> {losses[-1]:.4f}")
    return f"loss {losses[0]:.4f} -> {losses[-1]:.4f} | 10 steps | grads finite"


def check_lora() -> str:
    """An adapter attached, trained for ten steps, saved and loaded back.

    This is the shape the project needs, on a model small enough to run in a second: if peft
    and transformers are not in the image, this check is what says so.
    """
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import Qwen3Config, Qwen3ForCausalLM

    config = Qwen3Config(
        vocab_size=512,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=256,
    )
    base = Qwen3ForCausalLM(config).cuda().to(torch.bfloat16)
    model = get_peft_model(
        base,
        LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], bias="none"),
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    optimiser = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-3)
    torch.manual_seed(0)
    ids = torch.randint(0, 512, (4, 32), device="cuda")
    losses = []
    for _ in range(10):
        out = model(input_ids=ids, labels=ids)
        out.loss.backward()
        optimiser.step()
        optimiser.zero_grad()
        losses.append(out.loss.item())
    torch.cuda.synchronize()
    if losses[-1] >= losses[0]:
        raise AssertionError(f"loss did not fall: {losses[0]:.4f} -> {losses[-1]:.4f}")
    path = "/work/probe-adapter"
    model.save_pretrained(path)
    reloaded = PeftModel.from_pretrained(Qwen3ForCausalLM(config).cuda().to(torch.bfloat16), path)
    return (
        f"trainable {trainable:,}/{total:,} ({100 * trainable / total:.2f}%) | "
        f"loss {losses[0]:.2f} -> {losses[-1]:.2f} | reloaded {type(reloaded).__name__}"
    )


def check_checkpoint(name: str) -> str:
    """Load one checkpoint from the read-only cache and run a forward pass."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = time.time()
    tokeniser = AutoTokenizer.from_pretrained(name)
    tokenised = time.time()
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map={"": 0})
    model.eval()
    loaded = time.time()
    ids = tokeniser("2 + 2 =", return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model(**ids)
    torch.cuda.synchronize()
    forward = time.time() - loaded
    params = sum(p.numel() for p in model.parameters())
    peak = torch.cuda.max_memory_allocated() / 2**30
    return (
        f"{name} | tokeniser {tokenised - started:.1f}s load {loaded - tokenised:.1f}s "
        f"forward {forward:.2f}s | logits={tuple(out.logits.shape)} params={params:,} peak={peak:.2f} GiB"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--checkpoint", default=None, help="a checkpoint to load and run once")
    parser.add_argument(
        "--require-sm121",
        action="store_true",
        help="fail unless sm_121 is in the compiled architecture list (default: accept PTX JIT)",
    )
    args = parser.parse_args(argv)

    import torch

    print("torch     ", torch.__version__, "| built for CUDA", torch.version.cuda)
    print("python    ", sys.version.split()[0])
    try:
        from peft import __version__ as peft_version
        from transformers import __version__ as transformers_version

        print("transformers", transformers_version, "| peft", peft_version)
    except Exception as exc:  # noqa: BLE001
        print("transformers/peft: NOT AVAILABLE -", type(exc).__name__)
    print()

    ok = True
    ok &= _timed("device + archs", check_device)
    if args.require_sm121 and "sm_121" not in torch.cuda.get_arch_list():
        print("FAIL  sm_121 required but not compiled in")
        ok = False
    ok &= _timed("fp32 exactness", check_fp32_exactness)
    ok &= _timed("bf16 matmul", check_bf16_matmul)
    ok &= _timed("throughput", check_throughput)
    ok &= _timed("sdpa (causal)", check_sdpa)
    ok &= _timed("backward + AdamW", check_backward)
    ok &= _timed("lora attach/save", check_lora)
    if args.checkpoint:
        ok &= _timed("checkpoint forward", lambda: check_checkpoint(args.checkpoint))

    print()
    print("VERDICT:", "USABLE" if ok else "NOT USABLE")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
