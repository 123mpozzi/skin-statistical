import unittest
import os


code_dir = 'code'

def set_working_dir(test: unittest.TestCase):
    '''Make the tests start with working dir =  code directory'''
    if os.path.basename(os.getcwd()) != code_dir:
        os.chdir("..")
    print('testing get_cwd(): ')
    current_dir = os.getcwd()
    print(current_dir)
    test.assertIsNotNone(current_dir)
    test.assertEqual(os.path.basename(current_dir), code_dir)
