import os
from utils.Schmugge import Schmugge
from utils.skin_dataset import SingletonMeta

class medium(Schmugge, metaclass=SingletonMeta):
    def __init__(self):
        super().__init__('medium')
        self.csv = os.path.join(self.dir, 'medium.csv')
    
    def get_all_paths(self) -> list:
        return super().get_test_paths()
