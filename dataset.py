from torch.utils.data import Dataset

class ChessDataset(Dataset):

    def __init__(self, X, v, p):
        self.X = X
        self.v = v
        self.p = p
      

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.v[idx], self.p[idx]