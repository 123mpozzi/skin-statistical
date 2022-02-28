import os
from utils.Schmugge import Schmugge
from utils.skin_dataset import SingletonMeta

class dark(Schmugge, metaclass=SingletonMeta):
    def __init__(self):
        super().__init__('dark')
        self.csv = os.path.join(self.dir, 'dark.csv')
    
    def get_all_paths(self) -> list:
        return super().get_test_paths()
