import unittest
import os


def set_working_dir(test: unittest.TestCase):
    '''Make the tests start at the project working dir'''
    os.chdir("..")
    print('testing get_cwd()')
    current_dir = os.getcwd()
    test.assertIsNotNone(current_dir)
    test.assertEqual(current_dir, 'code')

