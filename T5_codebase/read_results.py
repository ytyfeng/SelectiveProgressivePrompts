import numpy as np

def read_results():
    path = './save_dir/T5_experiment/results_dict.npy'
    results_dict = np.load(path, allow_pickle=True).item()
    print(results_dict)

if __name__ == "__main__":
    read_results()
