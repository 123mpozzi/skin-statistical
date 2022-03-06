import unittest
import os
from utils.logmanager import *


code_dir = 'code'

def set_working_dir(test: unittest.TestCase):
    '''Make the tests start with working dir = code directory'''
    if os.path.basename(os.getcwd()) != code_dir:
        os.chdir("..")
    current_dir = os.getcwd()
    info('testing get_cwd(): ' + current_dir)
    test.assertIsNotNone(current_dir)
    test.assertEqual(os.path.basename(current_dir), code_dir)
