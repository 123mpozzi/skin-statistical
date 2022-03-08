import os
import shutil
import unittest

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

def search_subdir(root_dir: str, target_name: str) -> str:
    '''
    Search inside a directory for the most recent matching subdirectory
    and return it or None if not found
    '''
    match = None
    # Fetch most recent prediction folder with the same name as pred
    for root, subdirs, files in os.walk(os.path.abspath(root_dir)):
        for d in subdirs:
            if d == target_name:
                d_abs = os.path.join(root, d)
                if match is None:
                    match = d_abs
                    info('Folder match found: ' + match)
                else:
                    # get the latest modified folder
                    if os.path.getmtime(d_abs) > os.path.getmtime(match):
                        match = d_abs
                        info('Folder match updated: ' + match)
    return match


def rm_folder(folder: str):
    if os.path.isdir(folder):
        shutil.rmtree(folder)
