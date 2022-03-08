import os
import unittest

from cli.multipredict import batch_multi, single_multi
from click.testing import CliRunner
from predict import predictions_dir
from utils.db_utils import gen_pred_folders, get_db_by_name, get_models
from utils.hash_utils import hash_dir
from utils.logmanager import *
from utils.Schmugge import medium, light

from tests.helper import search_subdir, set_working_dir

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

class TestMultipredict(unittest.TestCase):
    '''Functional testing for multipredict commands'''


    def check_predictions_folders(self, predictions: list):
        '''
        Resulting predictions have same hashes as the ones registered in the thesis,
        for datasets featured in it

        Prediction images are the same number as in the csv file
        '''
        for pred in predictions:
            match = search_subdir(predictions_dir, pred)
            # Assert predictions folder exists
            self.assertIsNotNone(match, f'No match found for prediction folder named {pred}')

            # for datasets featured in thesis
            if pred in hashes:
                match_hash = hash_dir(match)
                # Resulting predictions have same hashes
                self.assertEqual(match_hash, hashes[pred])
                info('Hash corresponding for ' + pred)
            
            #  Predictions folder it has same number of files as defined in csv
            target = pred.split('_on_')[1]
            target = get_db_by_name(target)
            # base pred or cross pred (count test paths or all paths) ?
            base_pred = True if pred.split('_on_')[1] == pred.split('_on_')[0] else False
            images_to_predict = len(target.get_all_paths())
            if base_pred:
                images_to_predict = len(target.get_test_paths())
            images_predicted = len(os.listdir(os.path.join(match, 'p'))) # images in prediction dir
            self.assertEqual(images_predicted, images_to_predict,
                f'Number of images predicted != number of images to predict: {images_predicted} != {images_to_predict}')

    def test_batchm(self): # TODO: not working..
        '''
        Command run without errors

        Resulting predictions have same hashes as the ones registered in the thesis,
        for datasets featured in it.

        Prediction images are the same number as in the csv file
        '''
        set_working_dir(self)

        runner = CliRunner()

        predictions = gen_pred_folders(get_models(), 'all')

        datasets_args = []
        # NOTE: uncomment to test all datasets (long time)
        #for m in get_models():
        for m in [medium(), light()]: # get models
            datasets_args.append('-t')
            datasets_args.append(m.name)

        info('TESTING COMMANDS...')
        # run command
        result = runner.invoke(batch_multi, ['-m', 'all'].extend(datasets_args))
        # Command has no errors on run
        self.assertEqual(result.exit_code, 0,
            f'Error running the command with "-m all {" ".join(datasets_args)}"\nResult: {result}')
        
        info('TESTING RESULTING PREDICTIONS...')
        self.check_predictions_folders(predictions)

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
        info('TESTING COMMANDS...')
        # NOTE: uncomment to test all datasets (long time)
        #for m in get_models(): # get models
        #    for p in get_datasets(): # get targets
        for m in [medium()]:
            for p in [medium(), light()]:
                # save runned prediction in the list
                predictions.append(f'{m.name}_on_{p.name}') # TODO: debug: print just command list, or this list
                # run command
                result = runner.invoke(single_multi, ['-m', m.name, '-p', p.name])
                # Command has no errors on run
                self.assertEqual(result.exit_code, 0,
                    f'Error running the command with "-m {m.name} -p {p.name}"\nResult: {result}')
        print(predictions)
        
        info('TESTING RESULTING PREDICTIONS...')
        self.check_predictions_folders(predictions)


if __name__ == '__main__':
    unittest.main()
