import os

from utils.skin_dataset import SingletonMeta, skin_dataset


class VPU(skin_dataset, metaclass=SingletonMeta):
    '''
    VPU (2013), as for Video Processing & Understanding Lab, consists of 285 images
    taken from five different public datasets for human activity recognition.
    The size of the pictures is constant between the images of the same origin.
    The dataset provides native train and test splits. It is also referred to as VDM.
    `http://www-vpu.eps.uam.es/publications/SkinDetDM/#dataset`
    ---
    SanMiguel, J. C., & Suja, S. (2013). Skin detection by dual maximization of
    detectors agreement for video monitoring. Pattern Recognition Letters, 34(16),
    2102-2109.
    https://doi.org/10.1016/j.patrec.2013.07.016
    '''
    def __init__(self):
        super().__init__('VPU')
    
    def get_parts(self) -> list:
        return (ED_train(), ED_test(), LIRIS_train(), LIRIS_test(), SSG_train(), SSG_test(),
        UT_train(), UT_test(), AMI_train(), AMI_test())
    
    # Note: no need to use argument 'predefined' as this dataset has already defined splits
    def reset(self, predefined) -> str:
        trace = ''

        try:
            os.remove(self.csv)
        except OSError:
            pass

        for db in self.get_parts():
            newtrace = db.reset(append=True)
            if newtrace: # if string is not empty
                trace = trace + newtrace + '\n'
        
        return trace
    
class ED_test(skin_dataset, metaclass=SingletonMeta):
    def __init__(self):
        super().__init__(VPU().name)
        self.name = 'ED_test'
        self.gt, self.ori, self.new_gt = vpu_paths(self)
        self.gt_process = 'skin=0_0_255'
        self.note = self.nt_testing

class ED_train(skin_dataset, metaclass=SingletonMeta):
    def __init__(self):
        super().__init__(VPU().name)
        self.name = 'ED_test'
        self.gt, self.ori, self.new_gt = vpu_paths(self)
        self.gt_process = 'skin=0_0_255'
        self.ori_format = 'img(.*)'
        self.note = self.nt_training

class LIRIS_test(skin_dataset, metaclass=SingletonMeta):
    def __init__(self):
        super().__init__(VPU().name)
        self.name = 'LIRIS_test'
        self.gt, self.ori, self.new_gt = vpu_paths(self)
        self.gt_process = 'skin=0_0_255'
        self.note = self.nt_testing

class LIRIS_train(skin_dataset, metaclass=SingletonMeta):
    def __init__(self):
        super().__init__(VPU().name)
        self.name = 'LIRIS_train'
        self.gt, self.ori, self.new_gt = vpu_paths(self)
        self.gt_process = 'skin=0_0_255'
        self.note = self.nt_training

class SSG_test(skin_dataset, metaclass=SingletonMeta):
    def __init__(self):
        super().__init__(VPU().name)
        self.name = 'SSG_test'
        self.gt, self.ori, self.new_gt = vpu_paths(self)
        self.gt_process = 'skin=0_0_255'
        self.note = self.nt_testing
        self.gt_format = '(.*)_an'
        self.ori_format = '(.*)_raw'

class SSG_train(skin_dataset, metaclass=SingletonMeta):
    def __init__(self):
        super().__init__(VPU().name)
        self.name = 'SSG_train'
        self.gt, self.ori, self.new_gt = vpu_paths(self)
        self.gt_process = 'skin=0_0_255'
        self.note = self.nt_training

class UT_test(skin_dataset, metaclass=SingletonMeta):
    def __init__(self):
        super().__init__(VPU().name)
        self.name = 'UT_test'
        self.gt, self.ori, self.new_gt = vpu_paths(self)
        self.gt_process = 'skin=0_0_255'
        self.note = self.nt_testing

class UT_train(skin_dataset, metaclass=SingletonMeta):
    def __init__(self):
        super().__init__(VPU().name)
        self.name = 'UT_train'
        self.gt, self.ori, self.new_gt = vpu_paths(self)
        self.gt_process = 'skin=0_0_255'
        self.note = self.nt_training

class AMI_test(skin_dataset, metaclass=SingletonMeta):
    def __init__(self):
        super().__init__(VPU().name)
        self.name = 'AMI_test'
        self.gt, self.ori, self.new_gt = vpu_paths(self)
        self.gt_process = 'skin=0_0_255'
        self.note = self.nt_testing

class AMI_train(skin_dataset, metaclass=SingletonMeta):
    def __init__(self):
        super().__init__(VPU().name)
        self.name = 'AMI_train'
        self.gt, self.ori, self.new_gt = vpu_paths(self)
        self.gt_process = 'skin=0_0_255'
        self.note = self.nt_training


def vpu_paths(db: skin_dataset) -> list:
    dbpart, nt = db.name.split('_')
    middle_path = nt + '_' + dbpart

    gt_dir = os.path.join(db.dir, nt, middle_path, 'ann')
    ori_dir = os.path.join(db.dir, nt, middle_path, 'raw')
    new_gt_dir = os.path.join(db.dir, 'vpu_newskin', middle_path)
    return gt_dir, ori_dir, new_gt_dir
