from torch.utils.data import Dataset, DataLoader
from configs import CONFIGS
import os 
import torch


class CompletionDataset(Dataset):

    def __init__(self, root, train_test):
        super(Dataset, self).__init__()
        self.root = root
        self.dir = self.root + '/' + train_test
        if train_test == 'train':
            self.take = CONFIGS.dataset.take_train
        elif train_test == 'test':
            self.take = CONFIGS.dataset.take_test
        self.filenames = os.listdir(self.dir)[0: self.take]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        out = torch.load(self.dir + '/' + self.filenames[idx])
        return out

if __name__ == '__main__':
    from configs import CONFIGS
    from torch.utils.data import DataLoader
    dataset = CompletionDataset(CONFIGS.DATA.dataset.data_process_dir , 'train')
    data_loader = DataLoader(dataset, batch_size=2)
    for i, data in enumerate(data_loader):
        print(data['x'].dtype)
        if i == 0:
            break
