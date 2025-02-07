import torch
from torch.utils.data import Dataset, DataLoader


class TaskDataset(Dataset):
    def __init__(self, num_samples, vocab_size, max_length, num_classes):
        self.num_samples = num_samples
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        task_description = torch.randint(0, self.vocab_size, (self.max_length,))
        architecture = torch.rand((10, self.num_classes))  # 10 слоев
        weights = torch.rand((10, self.num_classes))  # 10 слоев
        quality_metric = torch.rand(1)
        
        return task_description, architecture, weights, quality_metric
