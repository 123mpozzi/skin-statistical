import unittest

from cli.training import train
from click.testing import CliRunner
from utils.db_utils import get_model_filename
from utils.hash_utils import hash_file
from utils.logmanager import *
from utils.Schmugge import medium

from tests.helper import set_working_dir

# xxh3_64 hashes of models already generated for thesis
hashes = {
    'ECU' : '29a2169cae186445',
    'HGR_small' : '08701a55b60f2009',
    'Schmugge' : '1a022f5df38f6e68',
    #'dark' : '080e3d4bd5f0d91c', # not reproducible: augmented images
    'medium' : '3a6b793973d8bed7',
    'light' : '87d32a50896bac76',
}


class TestTrain(unittest.TestCase):
    '''Functional testing for training commands'''

    def test_train(self):
        '''
        Command run without errors

        Model file hash is the same for one of the datasets featured in the thesis
        '''
        set_working_dir(self)

        runner = CliRunner()

        info('TESTING COMMANDS...')
        # NOTE: uncomment to test all trainable datasets
        #for d in get_trainable():
        for d in [medium()]:
            result = runner.invoke(train, ['-d', d.name])
            # Command has no errors on run
            self.assertEqual(result.exit_code, 0,
                f'Error running train command with "-d {d.name}"\nResult: {result}')
        
        info('TESTING CSV CONTENT...')
        #for d in get_trainable():
        for d in [medium()]:
            if d.name in hashes:
                model_name = get_model_filename(d)
                info('Testing model named ' + model_name)
                self.assertEqual(hash_file(model_name), hashes[d.name])


if __name__ == '__main__':
    unittest.main()
