import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
    
def greedy_selection(
    features,
    k,
    dSmax,
    prev_selected_features=None,
    prev_selected_dSmax=None,
    init_frame_dSmax=None,
    struc_tol=0.005,
    seed=None,
):
    """
    Greedy diversity selection using max-min distance in feature space.

    Args:
        features: Normalized feature array (num_frames, n_features)
        k: Number of samples to select
        dSmax: dSmax values for current bin
        prev_selected_features: Features of previously selected frames from prior bins
        prev_selected_dSmax: dSmax values of previously selected frames from prior bins
        init_frame_dSmax: dSmax value of frame 0 (input frame) to enforce distance from
        struc_tol: Minimum dSmax difference threshold (default: 0.005)
        seed: Random seed for reproducibility (default: None)

    Returns:
        Array of selected indices
    """
    if seed is not None:
        np.random.seed(seed)

    num_frames = len(features)
    selected = []

    # Select the first frame for this bin
    if prev_selected_features is not None and len(prev_selected_features) > 0:
        # If we have previously selected frames, start by finding the frame
        # that maximizes distance from all previously selected frames
        max_min_dist = -1
        best_idx = None

        for i in range(num_frames):
            # Check dSmax constraint against frame 0
            if init_frame_dSmax is not None and abs(dSmax[i] - init_frame_dSmax) <= struc_tol:
                continue

            # Check dSmax constraint against previously selected frames
            dSmax_valid = all(abs(dSmax[i] - prev_dSmax) > struc_tol for prev_dSmax in prev_selected_dSmax)
            if not dSmax_valid:
                continue

            # Compute minimum distance to all previously selected frames
            distances_to_prev = [np.linalg.norm(features[i] - prev_feat) for prev_feat in prev_selected_features]
            min_dist = np.min(distances_to_prev)

            if min_dist > max_min_dist:
                max_min_dist = min_dist
                best_idx = i

        if best_idx is not None:
            selected.append(best_idx)
        else:
            return np.array(selected)
    else:
        # No previous frames and no must_include_idx: start with a random point
        selected.append(np.random.randint(0, num_frames))

    # Pick remaining frames by greedily adding points that maximize minimum distance to selected points
    for _ in range(k - 1):
        max_min_dist = -1
        best_idx = None

        # find frame that maximizes diversity from previously selected frames
        for i in range(num_frames):
            if i in selected:
                continue

            # Check dSmax constraint against frame 0
            if init_frame_dSmax is not None and abs(dSmax[i] - init_frame_dSmax) <= struc_tol:
                continue

            # Check dSmax constraint: all differences must be > struc_tol
            # Check against current bin's selected frames
            dSmax_valid = all(abs(dSmax[i] - dSmax[j]) > struc_tol for j in selected)

            # Also check against previously selected frames from prior bins
            if dSmax_valid and prev_selected_dSmax is not None:
                dSmax_valid = all(abs(dSmax[i] - prev_dSmax) > struc_tol for prev_dSmax in prev_selected_dSmax)

            if not dSmax_valid:
                continue

            # Compute minimum distance from current frame to selected frames in current bin
            distances_to_selected = [np.linalg.norm(features[i] - features[j]) for j in selected]

            # Also compute distances to previously selected frames from prior bins
            if prev_selected_features is not None and len(prev_selected_features) > 0:
                distances_to_prev = [np.linalg.norm(features[i] - prev_feat) for prev_feat in prev_selected_features]
                all_distances = distances_to_selected + distances_to_prev
            else:
                all_distances = distances_to_selected

            min_dist = np.min(all_distances)

            # Take the maximum min distance among all frames
            if min_dist > max_min_dist:
                max_min_dist = min_dist
                best_idx = i

        if best_idx is not None:
            selected.append(best_idx)

    return np.array(selected)

def select_by_force(dE, Fmax, dSmax, frame_idx, bin_count, bins, struc_tol=0.005, seed=None):
    """
    Select frames based on force binning strategy.

    Args:
        dE: Energy difference values
        Fmax: Maximum force values
        dSmax: Maximum structural displacement values
        frame_idx: Frame indices
        bin_count: Tuple of (b1, b2, b3, b4) - number of samples per bin
        bins: List of tuples (lower, upper) defining bin edges
        struc_tol: Minimum dSmax difference threshold (default: 0.005)
        seed: Random seed for reproducibility (default: None)

    Example bins:
    - Bin 1: F < 0.05 -> select up to k=3 (must include dSmax=0)
    - Bin 2: 0.05 <= F < 0.10 -> select up to k=2 (diversity)
    - Bin 3: 0.10 <= F < 0.30 -> select k=1 (random)
    - Bin 4: 0.30 <= F < 0.60 -> select k=1 (random)

    Returns:
        selected_indices: Indices of selected frames
        bin_labels: Which bin each selected frame came from
    """
    if seed is not None:
        np.random.seed(seed)

    # Find frame 0 (input frame) and get its dSmax value
    frame_zero_idx = np.where(frame_idx == 0)[0]
    if len(frame_zero_idx) > 0:
        frame_zero_idx = frame_zero_idx[0]
        init_frame_dSmax = dSmax[frame_zero_idx]
    else:
        # If frame 0 doesn't exist, no constraint
        frame_zero_idx = None
        init_frame_dSmax = None

    # Normalize all features globally for consistent feature space across bins
    all_features = np.column_stack([dE, Fmax, dSmax])
    scaler = MinMaxScaler()
    all_features_norm = scaler.fit_transform(all_features)

    # Get sample counts per bin
    b1, b2, b3, b4 = bin_count
    bin_counts = [b1, b2, b3, b4]

    # Find the opt_frame (lowest dSmax frame) - this is the optimized/relaxed structure
    opt_frame_idx = np.argmin(dSmax)

    # Always select opt_frame first
    selected_indices = [opt_frame_idx]

    # Determine which bin opt_frame belongs to based on its Fmax value
    opt_frame_fmax = Fmax[opt_frame_idx]
    opt_frame_bin = None
    for bin_num, (lower, upper) in enumerate(bins, start=1):
        if lower <= opt_frame_fmax < upper:
            opt_frame_bin = bin_num
            break

    # Initialize bin_labels with opt_frame's bin
    bin_labels = [opt_frame_bin] if opt_frame_bin is not None else []

    # Track previously selected frames across all bins (starting with opt_frame)
    prev_selected_features = [all_features_norm[opt_frame_idx]]
    prev_selected_dSmax = [dSmax[opt_frame_idx]]

    for bin_num, ((lower, upper), k) in enumerate(zip(bins, bin_counts), start=1):
        # Get indices in this bin
        bin_mask = (Fmax >= lower) & (Fmax < upper) # [0, 1, 1, 0, ..., 1]
        bin_indices = np.where(bin_mask)[0] # return indicies

        # Exclude frame 0 from all bins (it's the input frame, not a training target)
        if frame_zero_idx is not None:
            bin_indices = bin_indices[bin_indices != frame_zero_idx]

        # Exclude opt_frame from all bins (already selected)
        bin_indices = bin_indices[bin_indices != opt_frame_idx]

        if len(bin_indices) == 0:
            continue

        # If this is the bin where opt_frame belongs, reduce k by 1 since we already selected it
        k_adjusted = k - 1 if bin_num == opt_frame_bin else k

        if k_adjusted <= 0:
            continue
        
        # Select frames based on bin number
        if bin_num in (1, 2):

            # Use globally normalized features for this bin
            features_norm = all_features_norm[bin_indices]

            # select frames
            local_selected = greedy_selection(
                features=features_norm,
                k=min(k_adjusted, len(bin_indices)),
                dSmax=dSmax[bin_indices],
                prev_selected_features=prev_selected_features,
                prev_selected_dSmax=prev_selected_dSmax,
                init_frame_dSmax=init_frame_dSmax,
                struc_tol=struc_tol / 2 if bin_num == 1 else struc_tol,
                seed=seed,
            )
            selected = bin_indices[local_selected] if len(local_selected) > 0 else []

        else:  # Bins 3 & 4: random selection
            # Random but enforce dS spacing vs everything already selected and frame 0
            feasible = []
            for i in bin_indices:
                # Check against frame 0
                if init_frame_dSmax is not None and abs(dSmax[i] - init_frame_dSmax) <= struc_tol:
                    continue
                # Check against previously selected frames
                if all(abs(dSmax[i] - ds) > struc_tol for ds in prev_selected_dSmax):
                    feasible.append(i)
            feasible = np.array(feasible, dtype=int)
            if len(feasible) == 0:
                continue
            if len(feasible) <= k_adjusted:
                selected = feasible
            else:
                selected = np.random.choice(feasible, size=k_adjusted, replace=False)

        if len(selected) > 0:
            selected_indices.extend(selected)
            bin_labels.extend([bin_num] * len(selected))

            # Add current bin's selected frames to history for next bins
            prev_selected_features.extend([all_features_norm[idx] for idx in selected])
            prev_selected_dSmax.extend([dSmax[idx] for idx in selected])

    return np.array(selected_indices), np.array(bin_labels)