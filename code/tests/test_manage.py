import unittest

from cli.manage import reset
from click.testing import CliRunner
from utils.db_utils import (skin_databases, skin_databases_names,
                            skin_databases_skintones)
from utils.logmanager import *
from utils.Schmugge import count_skintones
from utils.skin_dataset import skin_dataset

from tests.helper import set_working_dir


class TestReset(unittest.TestCase):
    '''Functional testing for manage commands'''

    def assert_same_size(self, csv: list, csv_import: list, m: skin_dataset):
        if m.name in skin_databases_names(skin_databases_skintones):
            self.assertEqual(len(csv), count_skintones(m, m.name, m.import_csv))
        else:
            self.assertEqual(len(csv), len(csv_import))
    
    def compare_content(self, csv:list, csv_import: list, m: skin_dataset, equality: bool):
        if m.name in skin_databases_names(skin_databases_skintones):
            # Get basenames
            csv_basenames = sorted([m.to_basenames(entry) for entry in csv])
            csv_basenames_import = [m.to_basenames(entry) for entry in csv_import]
            # Filter import_csv for skintone and sort
            csv_basenames_import = sorted([e for e in csv_basenames_import if str(m.split_csv_fields(e)[3]).startswith(m.name)])
        else:
            csv_basenames = sorted([m.to_basenames(entry) for entry in csv])
            csv_basenames_import = sorted([m.to_basenames(entry) for entry in csv_import])
        
        if equality:
            self.assertEqual(csv_basenames, csv_basenames_import)
        else:
            self.assertNotEqual(csv_basenames, csv_basenames_import)

    def test_reset_predefined(self):
        '''
        Command run without errors

        Each filename has same parameters as in import_csv
        
        Lines are db size
        '''
        set_working_dir(self)

        runner = CliRunner()

        info('TESTING COMMANDS...')
        for d in skin_databases:
            if d.import_csv is not None: # db has predefined splits
                result = runner.invoke(reset, ['-d', d.name, '--predefined'])
                # Command has no errors on run
                self.assertEqual(result.exit_code, 0,
                    f'Error running reset command with "-d {d.name} --predefined"\nResult: {result}')
        
        info('TESTING CSV CONTENT...')
        for d in skin_databases:
            if d.import_csv is not None:
                info('Testing db named ' + d.name)

                csv = d.read_csv()
                csv_import = d.read_csv(d.import_csv)

                self.assert_same_size(csv, csv_import, d)
                self.compare_content(csv, csv_import, d, equality=True)

    def test_reset(self):
        '''
        Command run without errors

        Not all filenames has same parameters as in import_csv

        Lines are db size
        '''
        set_working_dir(self)

        runner = CliRunner()

        info('TESTING COMMANDS...')
        for d in skin_databases:
            result = runner.invoke(reset, ['-d', d.name])
            # Command has no errors on run
            self.assertEqual(result.exit_code, 0,
                f'Error running reset command with "-d {d.name}"\nResult: {result}')
        
        info('TESTING CSV CONTENT...')
        for d in skin_databases:
            # Not all filenames has same parameters as in import_csv
            if d.import_csv is not None:
                info('Testing db named ' + d.name)
                csv = d.read_csv()
                csv_import = d.read_csv(d.import_csv)

                if d.name !=  'dark': # dark is data-augmented if not predefined
                    self.assert_same_size(csv, csv_import, d)
                else:
                    self.assertGreater(len(csv), 0)
                self.compare_content(csv, csv_import, d, equality=False)


if __name__ == '__main__':
    unittest.main()
