import json

#######################################################
                # Configure GemNet-OC 
#######################################################
model_cfg = {
    "num_targets": 1,
    "num_spherical": 7,
    "num_radial": 128,
    "num_blocks": 4,
    "emb_size_atom": 256,
    "emb_size_edge": 512,
    "emb_size_trip_in": 64,
    "emb_size_trip_out": 64,
    "emb_size_quad_in": 32,
    "emb_size_quad_out": 32,
    "emb_size_aint_in": 64,
    "emb_size_aint_out": 64,
    "emb_size_rbf": 16,
    "emb_size_cbf": 16,
    "emb_size_sbf": 32,
    "num_before_skip": 2,
    "num_after_skip": 2,
    "num_concat": 1,
    "num_atom": 3,
    "num_output_afteratom": 3,
    "num_atom_emb_layers": 2,
    "num_global_out_layers": 2,
    "flow_matching": True,
    "use_pbc": True,
    "scale_backprop_forces": False,
    "cutoff": 12.0,
    "cutoff_qint": 12.0,
    "cutoff_aeaint": 12.0,
    "cutoff_aint": 12.0,
    "max_neighbors": 30,
    "max_neighbors_qint": 8,
    "max_neighbors_aeaint": 20,
    "max_neighbors_aint": 1000,
    "enforce_max_neighbors_strictly": True,
    "rbf": {"name": "gaussian"},
    "rbf_spherical": None,
    "envelope": {"name": "polynomial", "exponent": 5},
    "cbf": {"name": "spherical_harmonics"},
    "sbf": {"name": "legendre_outer"},
    "extensive": True,
    "forces_coupled": True,
    "output_init": "HeOrthogonal",
    "activation": "silu",
    "scale_file": None,
    "quad_interaction": False,
    "atom_edge_interaction": True,
    "edge_atom_interaction": True,
    "atom_interaction": True,
    "scale_basis": True,
}

with open("gemnet_cfg.json", "w") as f:
    json.dump(model_cfg, f, indent=2)

#######################################################
                # Configure EquiformerV2 
#######################################################
    model_cfg = {
    "max_neighbors": 20,
    "max_radius": 12.0,
    "max_num_elements": 90,
    "num_layers": 8,
    "sphere_channels": 128,
    "attn_hidden_channels": 64,
    "num_heads": 8,
    "attn_alpha_channels": 64,
    "attn_value_channels": 16,
    "ffn_hidden_channels": 128,
    "norm_type": "layer_norm_sh",
    "lmax_list": [4],
    "mmax_list": [2],
    "grid_resolution": 18,
    "num_sphere_samples": 128,
    "edge_channels": 128,
    "use_atom_edge_embedding": True,
    "distance_function": "gaussian",
    "num_distance_basis": 512,
    "attn_activation": "silu",
    "use_s2_act_attn": False,
    "ffn_activation": "silu",
    "use_gate_act": False,
    "use_gridmlp": True,
    "alpha_drop": 0.1,
    "drop_path_rate": 0.1,
    "proj_drop": 0.0,
    "weight_init": "uniform",
}

with open("equiformerv2_cfg.json", "w") as f:
    json.dump(model_cfg, f, indent=2)
