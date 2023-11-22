from torch.utils.data import Dataset, DataLoader
from typing import Any, Callable, Iterable, TypeVar, Generic, List, Optional, Union

T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')
_worker_init_fn_t = Callable[[int], None]

class CustomDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        # Data should be a list of tuples (input, target)
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Custom DataLoader:
class CustomDataLoader(DataLoader):
    def __init__(self, dataset: Dataset[T_co], batch_size: Optional[int] = 1):
        super().__init__(dataset, batch_size=batch_size)

        # You can add any custom initialization here
        self.text = dataset['text']
    def getText(self):
        return self.text
    # Example: Overriding the __iter__ method
    def __iter__(self):
        # Implement the custom iteration logic here if needed
        for batch in super().__iter__():
            # You can add custom processing to each batch here
            yield batch