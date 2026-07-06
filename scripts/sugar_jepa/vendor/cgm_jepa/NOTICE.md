# Vendored from CGM-JEPA

Files in this directory (`encoder.py`, `embed.py`, `modules.py`, `mask_utils.py`) are copied, with only
import-path adjustments (`utils.foo` -> `.foo`, to work as a local package instead of the upstream repo's
top-level layout), from:

- Source: https://github.com/cruiseresearchgroup/CGM-JEPA
- Files: `models/encoder.py`, `utils/embed.py`, `utils/modules.py`, `utils/mask_utils.py`
- Commit: `ef18c28517ce103f45069dec2594e3774b0fda4` (master, fetched 2026-07-04)
- License: MIT (`LICENSE_UPSTREAM` in this directory is the upstream repo's license file, copied verbatim)
- Paper: Muhammad, Li, Salim, Metwally. "CGM-JEPA: Learning Consistent Continuous Glucose Monitor
  Representations via Predictive Self-Supervised Pretraining." arXiv:2605.00933 (2026).

Only these 4 files are vendored (not the full upstream repo) because they are the minimal
dependency-light subset needed to load and run the pretrained `Encoder` for inference/fine-tuning — pure
`torch`/`numpy`, no dependency on the upstream repo's pinned `torch<2.7`/`transformers==4.33.3` baseline
requirements, which conflict with this project's `torch>=2.9.1`. `pretrain/`, `eval/`, and baseline-model
code (MOMENT, Mantis, TS2Vec, GluFormer) from the upstream repo are intentionally not vendored.

Pretrained weights (`cgm_jepa/config.json` + `cgm_jepa/model.safetensors`, ~4MB, MIT license, from
`CRUISEResearchGroup/CGM-JEPA` on Hugging Face Hub) are vendored locally under
`scripts/sugar_jepa/pretrained/cgm_jepa/` rather than fetched at runtime. This was originally a
workaround for a Windows TLS problem on this dev machine — since root-caused and fixed in
`scripts/common/network.py` (`apply_windows_tls_workarounds()`, called from `JepaEncoderWrapper` before
`Encoder.from_pretrained(...)`), so Hub-id loading (`--jepa-weights-dir CRUISEResearchGroup/CGM-JEPA`,
which `Encoder.from_pretrained` accepts the same way it accepts a local directory) now works fine on this
machine too. The vendored copy stays the default regardless — it's already fast and needs no network at
all, so there's no reason to prefer a Hub fetch for this proof of concept. See `scripts/common/network.py`
for the full explanation of the two TLS issues found (an `OPENSSL_Uplink` process crash caused by
antivirus-injected `SSLKEYLOGFILE`, and a certificate-verification failure from the same antivirus's
HTTPS-inspection root CA not being in Python's bundled CA list). Reference config for this checkpoint (for
documentation only — not read by any code, the vendored `config.json` is what's actually loaded):

```json
{
  "attn_drop_rate": 0.0, "dim_in": 12, "drop_rate": 0.0, "embed_bias": true, "embed_dim": 96,
  "jepa": false, "kernel_size": 3, "mlp_ratio": 4.0, "nhead": 6, "num_layers": 3,
  "qk_scale": null, "qkv_bias": true, "time_inp_dim": 5
}
```

Note: despite the encoder's `forward(self, x, ...)` docstring comment saying `x: (B, C, T)`, the actual
runtime shape (confirmed by reading `ValueEmbedding.forward`, which unpacks `x.shape` as
`(batch_size, num_patches, patch_length)`) is `(B, num_patches, patch_length)` with `patch_length == dim_in
== 12`. `num_patches` is not baked into any layer shape, so any window length that's a multiple of 12
works; the pretrained model saw 24 patches (288 steps / 24h) during training.
