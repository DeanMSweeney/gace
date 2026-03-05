# TransformerModule Walkthrough

This document explains how `TransformerModule` works in:

- `src/adsflow/models/TFT/transformer/transformer_module.py`

## High-Level Role

`TransformerModule` is the dense token backbone used by TFT-style flow matching.

It takes padded multimodal atom tokens plus conditioning, runs transformer attention, and returns:

- modality predictions: atom types, Cartesian positions, fractional coordinates, lattice lengths/angles
- auxiliary predictions: global property, global energy, atomic forces

## Input Contract

`forward(...)` expects:

- `x` as `(atom_types, pos, frac_coords, lengths_scaled, angles_radians)`
  - `atom_types`: `(B, M, V)` one-hot
  - `pos`: `(B, M, 3)`
  - `frac_coords`: `(B, M, 3)`
  - `lengths_scaled`: `(B, 1, 3)`
  - `angles_radians`: `(B, 1, 3)`
- `t` as one scalar per modality per sample: `(B,)` for each modality
- `feats` with at least:
  - `dataset_idx`, `spacegroup`, `token_is_periodic`
  - `charge`, `spin` if energy/force aux heads are active
- `padding_mask`: `(B, M)` with `True` for padded tokens

## `__init__` Structure

Main components set up in `__init__`:

1. Embedding layers
   - atom type embedding
   - linear projections for `pos`, `frac_coords`, `lengths_scaled`, `angles_radians`
   - optional sinusoidal sequence positional encoding
   - Fourier time encoding

2. Transformer trunk
   - `ModernTransformer` or `Transformer` based on `implementation`
   - optional RoPE configuration

3. Output heads
   - per-modality heads for atom type / positions / lattice channels

4. Optional cross-attention decoders
   - one cross-attn block per modality head

5. Auxiliary task heads
   - `global_property`, `global_energy`, `atomic_forces`
   - optional additional aux transformer blocks

## Forward Pass Flow

1. Unpack tensors and read conditioning fields.

2. Apply periodicity gating:
   - periodic tokens use `frac_coords`, `lengths_scaled`, `angles_radians`
   - non-periodic tokens use `pos`
   - optional `mask_material_coords` zeroes Cartesian coords for periodic tokens

3. Build masks:
   - `real_mask = 1 - padding_mask.int()`
   - `global_mask` for pooled outputs

4. Embed each modality and condition:
   - atom types via embedding table (`argmax` on one-hot)
   - continuous channels via linear layers
   - time/dataset/spacegroup embeddings combined as conditioning

5. Combine embeddings:
   - either sum all embeddings
   - or concatenate then project with `combine_input`

6. Run transformer trunk attention with `padding_mask`.

7. Decode modality outputs:
   - with cross-attention if enabled, otherwise directly from trunk output

8. Decode auxiliary outputs:
   - optional charge/spin conditioning for energy/forces
   - optional conservative-force mode via gradient of energy

9. Apply output masks and return:
   - modality predictions tuple
   - auxiliary predictions tuple

## `forward_with_cfg` (Classifier-Free Guidance)

`forward_with_cfg(...)` assumes batch layout is:

- first half: conditional
- second half: unconditional

It runs one combined forward pass and mixes outputs with:

- `uncond + cfg_scale * (cond - uncond)`

for each modality and auxiliary output.

## Notes For AdsFlow Integration

- Padding is required only for transformer mode; keep Equiformer mode unchanged.
- Use an adapter: sparse PyG batch -> dense padded tensors -> transformer -> unpad back to sparse.
- Current copied file still imports `zatom.*` modules; those imports must be rewired to local AdsFlow paths or copied dependencies.
