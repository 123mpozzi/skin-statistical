import unittest
import os
from utils.db_utils import get_db_by_name
from utils.skin_dataset import skin_dataset
from utils.db_utils import skin_databases, get_models
from utils.hash_utils import hash_dir
from cli.multipredict import single_multi, batch_multi
from click.testing import CliRunner
from predict import predictions_dir
from helper import set_working_dir

class TestMultipredict(unittest.TestCase):

    # xxh3_64 hashes of prediction folders already generated for thesis
    hashes = {
        # base
        'ECU_on_ECU' : 'e1dc03ee2bfbb903',
        'HGR_small_on_HGR_small' : '40752ebb0f0b4410',
        'Schmugge_on_Schmugge' : '860c1665a6a03c70',
        # cross
        'ECU_on_HGR_small' : 'eb81f2ba89db8b4d',
        'ECU_on_Schmugge' : '51f8f444f55e0b9f',
        'HGR_small_on_ECU' : '867b75fc6913665e',
        'HGR_small_on_Schmugge' : '91c2e16def9c552c',
        'Schmugge_on_ECU' : 'd35c57d06d621c06',
        'Schmugge_on_HGR_small' : 'f81c14afc0cf244e',
        # skintone base
        'dark_on_dark' : 'f1db4259767f19ed',
        'light_on_light' : 'adb974e9c9a49eb9',
        'medium_on_medium' : 'bf0d93cbc416c526',
        # skintone cross
        'dark_on_light' : '2452c528faa840eb',
        'dark_on_medium' : 'b3210cceb89379b1',
        'light_on_dark' : 'bf4eb7bfdefa3a37',
        'light_on_medium' : '1524f657dd4df6bc',
        'medium_on_dark' : 'a07c9a2c17242f7e',
        'medium_on_light' : 'b4cdb5d3a861b341'
    }

    #def set_working_dir(self):
    #    '''Make the tests start at the project working dir'''
    #    os.chdir("..")
    #    print('testing get_cwd()')
    #    current_dir = os.getcwd()
    #    self.assertIsNotNone(current_dir)
    #    self.assertEqual(current_dir, 'code')

    def test_batchm(self):
        '''
        Command run without errors

        Each filename has same parameters as in import_csv
        
        Lines are db size
        '''

    def test_singlem(self):
        '''
        Command run without errors

        Resulting predictions have same hashes as the one registered in the thesis,
        for datasets featured in it.

        Prediction images are the same number as in the csv file
        '''
        set_working_dir(self)

        runner = CliRunner()

        predictions = []
        print('TESTING COMMANDS...')
        for m in get_models(): # get models
            print(m.name)
            for p in skin_databases: # get targets
                # save runned prediction in the list
                predictions.append(f'{m.name}_on_{p.name}')
                # run command
                result = runner.invoke(single_multi, ['-m' + m.name, '-p' + p.name])
                print(result)
                # Command has no errors on run
                self.assertEqual(result.exit_code, 0,
                    f'Error running the command with "-m {m.name} -p {p.name}"')
        
        print('TESTING RESULTING PREDICTIONS...')
        for pred in predictions:
            print(pred)
            match = None
            # Fetch most recent prediction folder with the same name as pred
            for root, subdirs, files in os.walk(os.path.abspath(predictions_dir)):
                for d in subdirs:
                    if d == pred:
                        if match is None:
                            match = d
                        else:
                            # get the latest modified folder
                            if os.path.getmtime(d) > os.path.getmtime(match):
                                match = d
            # Assert predictions folder exists
            self.assertIsNotNone(match, f'No match found for prediction folder named {pred}')

            # for datasets featured in thesis
            if pred in self.hashes:
                match_hash = hash_dir(match)
                # Resulting predictions have same hashes
                self.assertEqual(match_hash, self.hashes[pred])
            
            #  Predictions folder it has same number of files as defined in csv
            target = pred.split('_on_')[1]
            target = get_db_by_name(target)
            # base pred or cross pred (count test paths or all paths) ?
            base_pred = True if pred.split('_on_')[1] == pred.split('_on_')[0] else False
            images_to_predict = len(target.get_all_paths())
            if base_pred:
                images_to_predict = len(target.get_test_paths())
            images_predicted = len(os.listdir(match))
            self.assertEqual(images_predicted, images_to_predict,
                f'Number of images predicted != number of images to predict: {images_predicted} != {images_to_predict}')


if __name__ == '__main__':
    unittest.main()
