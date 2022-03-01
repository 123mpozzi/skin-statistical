import os
from utils.Schmugge import Schmugge
from utils.skin_dataset import SingletonMeta

class dark(Schmugge, metaclass=SingletonMeta):
    def __init__(self):
        super().__init__('dark')
        self.csv = os.path.join(self.dir, 'dark.csv')
    
    def get_all_paths(self) -> list: # TODO: ? these datasets are sub-datasets and use the same CSV with different te/tr/va splits
        return super().get_test_paths()
