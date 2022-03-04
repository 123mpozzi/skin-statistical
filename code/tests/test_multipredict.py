import unittest
from utils.skin_dataset import skin_dataset
from utils.db_utils import skin_databases, skin_databases_names, skin_databases_skintones
from utils.Schmugge import count_skintones
from cli.multipredict import reset
from click.testing import CliRunner

class TestMultipredict(unittest.TestCase):

    def assert_same_size(self, csv: list, csv_import: list, m: skin_dataset):
        pass

    def test_single(self):
        '''
        Command run without errors

        Each filename has same parameters as in import_csv
        
        Lines are db size
        '''
        runner = CliRunner()

        print('TESTING COMMANDS...')
        for m in skin_databases:
            if m.import_csv is not None: # db has predefined splits
                print(m.name)
                result = runner.invoke(reset, ['-d' + m.name, '--predefined'])
                print(result)
                # Command has no errors on run
                assert result.exit_code == 0
        
        print('TESTING CSV CONTENT...')
        for m in skin_databases:
            if m.import_csv is not None:
                print(m.name)

                csv = m.read_csv()
                csv_import = m.read_csv(m.import_csv)

                self.assert_same_size(csv, csv_import, m)
                self.compare_content(csv, csv_import, m, equality=True)

    def test_batch(self):
        '''
        Command run without errors

        Not all filenames has same parameters as in import_csv

        Lines are db size
        '''
        runner = CliRunner()

        print('TESTING COMMANDS...')
        for m in skin_databases:
            print(m.name)
            result = runner.invoke(reset, ['-d' + m.name])
            print(result)
            # Command has no errors on run
            assert result.exit_code == 0
        
        print('TESTING CSV CONTENT...')
        for m in skin_databases:
            # Not all filenames has same parameters as in import_csv
            if m.import_csv is not None:
                print(m.name)
                csv = m.read_csv()
                csv_import = m.read_csv(m.import_csv)

                if m.name !=  'dark': # dark is data-augmented if not predefined
                    self.assert_same_size(csv, csv_import, m)
                self.compare_content(csv, csv_import, m, equality=False)


if __name__ == '__main__':
    unittest.main()
