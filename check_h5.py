import h5py

def print_h5_structure(file_path):
    with h5py.File(file_path, 'r') as f:
        print(f"File: {file_path}")
        print("\nDatasets:")
        for key in f.keys():
            print(f"- {key}: {f[key].shape}")
            print(f"  dtype: {f[key].dtype}")

# Check both h5 files
queries_h5 = "/Users/vigneshshanmugasundaram/Code/github/Search-Engine/data/ms_marco/msmarco_queries_dev_eval_embeddings.h5"
print_h5_structure(queries_h5)