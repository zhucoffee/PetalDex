# Changelog

## 0.1.0 — 2026-11-01

Initial community release, based on the source organized by Yue Xin (@vinkda).

- Add installation/data-format guidance, release limitations, citation metadata, contribution guidance, and issue/PR templates.
- Retain MIT notices for adapted Diffusion Policy utilities and separate code and dataset terms.
- Protect conversion input paths, publish completed HDF5/MCAP stores, stream MCAP conversion by bag, and reject non-finite joint values.
- Replace interactive sampler debugging with contextual errors and explicitly reject unsupported internal validation splitting.
- Fix DDP device access, checkpoint/optimizer restoration, EMA inference selection, and gradient accumulation boundaries.
- Retain the original dependency versions and package mirror; rename distribution metadata to PetalDex while retaining the `psi_policy` import namespace.

These changes were reviewed statically. Fresh environment installation, conversion, GPU training/inference, and hardware execution have not been performed for this release.
