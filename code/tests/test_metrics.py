import json
import unittest

import numpy as np
from cli.measure import dump_dir, eval
from click.testing import CliRunner
from metrics import *
from utils.logmanager import *

from tests.helper import set_working_dir

docs_dir = os.path.join('..', 'docs')
docs_y_path = os.path.join(docs_dir, 'y')
docs_p_path = os.path.join(docs_dir, 'p')


def load_sample_images() -> list:
    '''Load and return groundtruths and predictions of documentation images'''
    results = []
    for i in ('infohiding', 'st-vincent-actor-album-art'):
        gt_path = os.path.join(docs_y_path, f'{i}.png')
        pred_path = os.path.join(docs_p_path, f'{i}.png')
        gt_bool, pred_bool = load_images(gt_path, pred_path)
        results.append((gt_bool, pred_bool))


# Test on documentation sample images
docs_images = load_sample_images()

sample_results = [
    {
        # image 1
        "dprs": 0.304876812223106,
        "f1": 0.793928629097368,
        "f2": 0.7443561670380776,
        "iou": 0.6582766561345258,
        "iou_logical": 0.6582766561345258,
        "mcc": 0.7825427429917932,
        "p": "..\\docs\\p\\infohiding.png",
        "precision": 0.8930543446672479,
        "recall": 0.7146096157200162,
        "specificity": 0.991948540672997,
        "y": "..\\docs\\y\\infohiding.png",
    },
    {
        # image 2
        "dprs": 1.3336708225653098,
        "f1": 0.13122254358421426,
        "f2": 0.18615591953500293,
        "iou": 0.07021838963955183,
        "iou_logical": 0.07021838963955183,
        "mcc": -0.2938348302781793,
        "p": "..\\docs\\p\\st-vincent-actor-album-art.png",
        "precision": 0.08796124470142402,
        "recall": 0.2582217507239728,
        "specificity": 0.3702157506761177,
        "y": "..\\docs\\y\\st-vincent-actor-album-art.png",
    },
    {
        # averages
        "dprs_medium": 0.7906072068513502,
        "dprs": 0.8192738173942079,
        "f1_medium": 0.4884531684885388,
        "f1": 0.46257558634079116,
        "f2": 0.46525604328654024,
        "iou": 0.3642475228870388,
        "iou_logical": 0.3642475228870388,
        "mcc": 0.24435395635680693,
        "precision": 0.4905077946843359,
        "recall": 0.4864156832219945,
        "specificity": 0.6810821456745574
    }
]



class TestMetrics(unittest.TestCase):
    '''Functional and Unit testing for metrics measurements'''


    def metrics_unittesting(self):
        # singles
        y_true = np.array([[1,0,0],[0,1,0],[0,0,1]]) > 0 # > 0 to cast as bool
        y_pred = np.array([[1,0,0],[1,0,0],[0,0,0]]) > 0
        tp = 1
        tn = 5
        fp = 1
        fn = 2
        cs = confmat_scores(y_true, y_pred)
        pr = tp / (tp+fp) # 0.5
        re = tp / (tp+fn) # 0.33
        sp = tn / (tn+fp) # 0.83
        f1_ = 2*re*pr / (re+pr)
        overlap = y_true * y_pred
        union =   y_true | y_pred # with bitwise or it would work even without casting matrix as bool
        iou_ = overlap.sum() / (union.sum())
        a = (1 - pr)**2 # 0.25
        b = (1 - re)**2 # 0.4489
        c = (1 - sp)**2 # 0.0289
        dprs_ = math.sqrt(a + b + c)
        #           3               /  V        2       *    3      *     6     *    7
        mcc_ = (tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        print(cs)
        self.assertEqual(round(pr, 2), 0.5)
        self.assertEqual(round(pr, 2), round(precision(cs), 2), 'pr not equal to its function')
        self.assertEqual(round(re, 2), 0.33)
        self.assertEqual(round(re, 2), round(recall(cs), 2), 're not equal to its function')
        self.assertEqual(round(sp, 2), 0.83)
        self.assertEqual(round(sp, 2), round(specificity(cs), 2), 'sp not equal to its function')
        self.assertEqual(round(f1_, 2), 0.40)
        self.assertEqual(round(f1_, 2), round(f1(cs), 2), 'f1 not equal to its function')
        print(overlap.sum())
        print(union.sum())
        print(iou_)
        print(iou(cs))
        self.assertEqual(round(iou_, 2), 0.25)
        self.assertEqual(round(iou_, 2), round(iou(cs), 2), 'iou not equal to its function')
        self.assertEqual(round(iou_, 2), round(iou_logical(y_true, y_pred), 2), 'iou not equal to its function (alt)')
        self.assertEqual(round(dprs_, 2), 0.85)
        self.assertEqual(round(dprs_, 2), round(dprs(cs), 2), 'dprs not equal to its function')
        self.assertEqual(round(mcc_, 2), 0.19)
        self.assertEqual(round(mcc_, 2), round(mcc(cs), 2), 'mcc not equal to its function')
        # medium avg
        #metrics = [f1_medium, f1, f2, iou, iou_logical, dprs_medium, dprs, mcc, recall, precision, specificity]
        #rpd = pd_metrics(docs_y_path, docs_p_path, metrics)
        #res = print_pd_mean(rpd, metrics, desc='unit testing')
        pr_1 = 0.51
        pr_2 = 0.82
        pr_avg = (pr_1 + pr_2) /2   # 0.67
        re_1 = 0.61
        re_2 = 0.45
        re_avg = (re_1 + re_2) /2   # 0.53
        sp_1 = 0.14
        sp_2 = 0.62
        sp_avg = (sp_1 + sp_2) /2   # 0.38
        f1_med_avg = pr_avg * re_avg * 2 / (pr_avg + re_avg)
        a_ = (1 - pr_avg)**2 # 0.10
        b_ = (1 - re_avg)**2 # 0.22
        c_ = (1 - sp_avg)**2 # 0.38
        dprs_med_avg = math.sqrt(a_ + b_ + c_)
        self.assertEqual(round(f1_med_avg, 2), 0.59)
        self.assertEqual(round(f1_med_avg, 2), round(f1_medium(pr_avg, re_avg, sp_avg), 2), 'f1-medium not equal to its function')
        self.assertEqual(round(dprs_med_avg, 2), 0.85)
        self.assertEqual(round(dprs_med_avg, 2), round(dprs_medium(pr_avg, re_avg, sp_avg), 2), 'dprs-medium not equal to its function')


    def test_eval(self):
        '''
        Metrics functions are correct for simple numbers

        Command run without errors

        Metrics are correct in respect to documentation sample images
        '''
        set_working_dir(self)

        runner = CliRunner()

        info('METRICS UNIT TESTING...')
        self.metrics_unittesting()

        dump_singles = os.path.join(dump_dir, 'metrics_docs_singles.json')
        dump_avg = os.path.join(dump_dir, 'metrics_docs_average.json')
        docs_dir = os.path.join('..', 'docs')

        info('TESTING EVAL COMMAND...')
        result = runner.invoke(eval, ['-p', docs_dir, '-d'])
        # Command has no errors on run
        self.assertEqual(result.exit_code, 0,
            f'Error running reset command with "-p {docs_dir} -d"\nResult: {result}')
        
        info('TESTING ON DOCS IMAGES...')
        with open(dump_singles) as json_file:
            singles = json.load(json_file)
            #singles = json.dumps(singles, indent=4)
        with open(dump_avg) as json_file:
            avg = json.load(json_file)
            #avg = json.dumps(avg, indent=4)
        self.assertIsNotNone(singles, 'Error, file not found: ' + dump_singles)
        self.assertIsNotNone(singles, 'Error, file not found: ' + dump_avg)
        singles.append(avg)
        for i in range(3):
            for key in singles[i]:
                self.assertEqual(singles[i][key], sample_results[i][key], 'Key value not equal: ' + key)


if __name__ == '__main__':
    unittest.main()
