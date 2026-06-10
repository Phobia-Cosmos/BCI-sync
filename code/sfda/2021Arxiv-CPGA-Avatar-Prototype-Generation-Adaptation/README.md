# CPGA Toy Privacy Probe

This directory is **not** the official CPGA implementation. I did not find a public official GitHub repository for:

`Source-free Domain Adaptation via Avatar Prototype Generation and Adaptation`

The script here is a minimal, reproducible probe for the security question:

> If source data is unavailable but the source model is released, can we recover source-like class prototypes from the model?

The toy experiment trains a simple source classifier on synthetic Gaussian source classes, discards the source data, and optimizes one avatar vector per class using only the frozen source model. If the generated avatars are close to the hidden source class centers, the source model has leaked class-level distribution information.

Run:

```bash
python3 cpga_toy_privacy_probe.py
```

Expected result: generated avatar prototypes should have high confidence under the source model and should be close to the original class centroids.

