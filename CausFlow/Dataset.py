import numpy as np
import torch
from torch.utils import data
import scanpy as sc

class Dataset(data.Dataset):
    def __init__(self, folder, profile_size, factor_list):
        super().__init__()
        self.folder = folder
        self.profile_size = profile_size
        data_obj = sc.read_h5ad(folder)
        X = data_obj.X[:, :profile_size]
        if not isinstance(X, np.ndarray):
            X = X.toarray()
        self.profile_data = X.astype(np.float32)

        # ======================
        # labels
        # ======================
        factor_dict = dict(zip(list(data_obj.obs.columns), range(len(list(data_obj.obs.columns)))))
        idx = [factor_dict[i] for i in factor_list]
        self.labels = data_obj.obs.iloc[:, idx]

        # ======================
        # weights
        # ======================
        tuple_list = []
        for i in range(len(data_obj)):
            tuple_list.append(str([data_obj.obs[j].iloc[i] for j in factor_list]))

        unique_tuple_list = np.unique(tuple_list)
        tmp_dict = dict(zip(unique_tuple_list, range(len(unique_tuple_list))))
        merged_class = [tmp_dict[i] for i in tuple_list]

        freq = {}
        for item in merged_class:
            freq[item] = freq.get(item, 0) + 1

        for i in freq:
            freq[i] = 1 - (freq[i] / len(data_obj))

        self.weights = np.array([freq[i] for i in merged_class], dtype=np.float32)

    def __len__(self):
        return self.profile_data.shape[0]

    def __getitem__(self, index):
        cell_exp = self.profile_data[index]
        cell_exp = torch.tensor(cell_exp, dtype=torch.float32)
        labels = np.array(list(self.labels.iloc[index, :]), dtype=int)
        labels = torch.tensor(labels, dtype=torch.long)
        weight = torch.tensor(self.weights[index], dtype=torch.float32)
        return cell_exp, labels, weight

# ==============================
# Generation Dataset
# ==============================
class Generation_Dataset(data.Dataset):
    def __init__(self, folder, profile_size, factor_list):
        super().__init__()
        self.folder = folder
        self.profile_size = profile_size
        data_obj = sc.read_h5ad(folder)
        X = data_obj.X[:, :profile_size]
        if not isinstance(X, np.ndarray):
            X = X.toarray()
        self.profile_data = X.astype(np.float32)

    def __len__(self):
        return self.profile_data.shape[0]

    def __getitem__(self, index):
        cell_exp = self.profile_data[index]
        cell_exp = torch.tensor(cell_exp, dtype=torch.float32)
        return cell_exp