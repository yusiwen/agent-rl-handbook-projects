# The training container for Project 2.
#
# Why an image instead of a virtualenv on the host: this box is aarch64 with a Blackwell Ultra
# GPU (GB10, compute capability 12.1), and the PyTorch wheels on PyPI are not built for it.
# NVIDIA ships an image that is, and it already carries triton, flash-attn and safetensors, so
# what is left to add is the two packages the supervised fine-tuning loop imports.
#
# The base is pinned by tag *and* digest: the tag names the release, the digest names the bytes.
# A tag alone moves when the vendor republishes the image.
#
# Every number in README.md was measured on this image. If a check fails after an image bump,
# the probe says which one and the README says what it looked like when it passed.

FROM nvcr.io/nvidia/pytorch:26.09-py3@sha256:6e8ccc607fc2a51e3741667b86316a0889418ba8b78ed0ba96e8d282b648a7a6

ENV HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false

# Pinned to the versions the probe was measured against. A range would make "it worked last
# month" unfalsifiable, which is the one thing a build recipe must not be.
RUN pip install --no-cache-dir \
        "transformers==5.20.1" \
        "peft==0.21.2"

WORKDIR /env
