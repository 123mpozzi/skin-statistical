import unittest
from utils.skin_dataset import skin_dataset
from utils.db_utils import skin_databases, skin_databases_names, skin_databases_skintones
from utils.Schmugge import count_skintones
from cli.manage import reset
from click.testing import CliRunner
from tests.helper import set_working_dir


class TestReset(unittest.TestCase):

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

        print('TESTING COMMANDS...')
        for m in skin_databases:
            if m.import_csv is not None: # db has predefined splits
                print(m.name)
                result = runner.invoke(reset, ['-d', m.name, '--predefined'])
                print(result)
                # Command has no errors on run
                self.assertEqual(result.exit_code, 0,
                    f'Error running reset command with "-d {m.name} --predefined"')
        
        print('TESTING CSV CONTENT...')
        for m in skin_databases:
            if m.import_csv is not None:
                print(m.name)

                csv = m.read_csv()
                csv_import = m.read_csv(m.import_csv)

                self.assert_same_size(csv, csv_import, m)
                self.compare_content(csv, csv_import, m, equality=True)

    def test_reset(self):
        '''
        Command run without errors

        Not all filenames has same parameters as in import_csv

        Lines are db size
        '''
        set_working_dir(self)

        runner = CliRunner()

        print('TESTING COMMANDS...')
        for m in skin_databases:
            print(m.name)
            result = runner.invoke(reset, ['-d', m.name])
            print(result)
            # Command has no errors on run
            self.assertEqual(result.exit_code, 0,
                f'Error running reset command with "-d {m.name}"')
        
        print('TESTING CSV CONTENT...')
        for m in skin_databases:
            # Not all filenames has same parameters as in import_csv
            if m.import_csv is not None:
                print(m.name)
                csv = m.read_csv()
                csv_import = m.read_csv(m.import_csv)

                if m.name !=  'dark': # dark is data-augmented if not predefined
                    self.assert_same_size(csv, csv_import, m)
                else:
                    self.assertGreater(len(csv), 0)
                self.compare_content(csv, csv_import, m, equality=False)


if __name__ == '__main__':
    unittest.main()
