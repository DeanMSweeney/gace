from adsflow.utils.data_processing import precompute_vel


if __name__ == "__main__":
    import time
    
    start = time.time()
    precompute_vel(
        src_path="/workspace/datasets/full/is2res_train_val_test_lmdbs/data/is2re/all/train/data.lmdb",
        dst_path="/workspace/datasets/full/is2res_train_val_test_lmdbs/data/is2re/all/train/data_refined.lmdb",
        batch_size=1,
        commit_every=5000,
    )
    end = time.time()
    
    print(f"Elapsed time: {end - start}")
