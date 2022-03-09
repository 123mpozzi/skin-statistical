import json
import unittest

import numpy as np
from cli.measure import dump_dir, eval
from click.testing import CliRunner
from metrics import *
from utils.logmanager import *
from utils.metrics_utils import load_images

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
        "dprs": "0.8193 \u00b1 0.51",
        "dprs_medium": "0.7906",
        "f1": "0.4626 \u00b1 0.33",
        "f1_medium": "0.4884",
        "f2": "0.4653 \u00b1 0.28",
        "iou": "0.3642 \u00b1 0.29",
        "iou_logical": "0.3642 \u00b1 0.29",
        "mcc": "0.2444 \u00b1 0.54",
        "method": "probabilistic",
        "precision": "0.4905 \u00b1 0.40",
        "recall": "0.4864 \u00b1 0.23",
        "specificity": "0.6811 \u00b1 0.31"
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
        info(cs)
        self.assertEqual(round(pr, 2), 0.5)
        self.assertEqual(round(pr, 2), round(precision(cs), 2), 'pr not equal to its function')
        self.assertEqual(round(re, 2), 0.33)
        self.assertEqual(round(re, 2), round(recall(cs), 2), 're not equal to its function')
        self.assertEqual(round(sp, 2), 0.83)
        self.assertEqual(round(sp, 2), round(specificity(cs), 2), 'sp not equal to its function')
        self.assertEqual(round(f1_, 2), 0.40)
        self.assertEqual(round(f1_, 2), round(f1(cs), 2), 'f1 not equal to its function')
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

    def mcc_unittest(self):
        '''
        Unit testing based on MCC's paper analysis
        
        The analysis shows how F1 doesn't care much about TN and could signal
        over-optimistic data to the classifier
        '''
        # Use Case A1: Positively imbalanced dataset
        data = {}
        data['ap'] = 91   # 91 sick patients
        data['an'] = 9    # 9 healthy individuals
        data['se'] = 99
        data['tp'] = 90   # algorithm is good at predicting positive data
        data['fp'] = 9
        data['tn'] = 0
        data['fn'] = 1    # algorithm is bad at predicting negative data
        # F1 measures an almost perfect score, MCC instead measures a bad score
        # F1 0.95    MCC -0.03
        f1_ = round(f1(data), 2)
        mcc_ = round(mcc(data), 2)
        self.assertEqual(f1_, 0.95)
        self.assertEqual(mcc_, -0.03)

        # Use Case A2: Positively imbalanced dataset
        data = {}
        data['ap'] = 75   # 75 positives
        data['an'] = 25   # 25 negatives
        data['se'] = 11
        data['tp'] = 5    # classifier unable to predict positives
        data['fp'] = 6
        data['tn'] = 19   # classifier was able to predict negatives
        data['fn'] = 70
        # In this case both the metrics measure a bad score
        # F1 0.12    MCC -0.24
        f1_ = round(f1(data), 2)
        mcc_ = round(mcc(data), 2)
        self.assertEqual(f1_, 0.12)
        self.assertEqual(mcc_, -0.24)

        # Use Case B1: Balanced dataset
        data = {}
        data['ap'] = 50   # 50 positives
        data['an'] = 50   # 50 negatives
        data['se'] = 92
        data['tp'] = 47   # classifier able to predict positives
        data['fp'] = 45
        data['tn'] = 5    # classifier was unable to predict negatives
        data['fn'] = 3
        # F1 measures a good score, MCC doesn't
        # F1 0.66    MCC 0.07
        f1_ = round(f1(data), 2)
        mcc_ = round(mcc(data), 2)
        self.assertEqual(f1_, 0.66)
        self.assertEqual(mcc_, 0.07)

        # Use Case B2: Balanced dataset
        data = {}
        data['ap'] = 50   # 50 positives
        data['an'] = 50   # 50 negatives
        data['se'] = 14
        data['tp'] = 10   # classifier was unable to predict positives
        data['fp'] = 4
        data['tn'] = 46    # classifier able to predict negatives
        data['fn'] = 40
        # F1 measures a good score, MCC doesn't
        # F1 0.31    MCC 0.17
        f1_ = round(f1(data), 2)
        mcc_ = round(mcc(data), 2)
        self.assertEqual(f1_, 0.31)
        self.assertEqual(mcc_, 0.17)

        # Use Case C1: Negatively imbalanced dataset
        data = {}
        data['ap'] = 10   # 10 positives
        data['an'] = 90   # 90 negatives
        data['se'] = 98
        data['tp'] = 9    # classifier was unable to predict positives
        data['fp'] = 89
        data['tn'] = 1    # classifier able to predict negatives
        data['fn'] = 1
        # Both the scores gives bad measure
        # F1 0.17    MCC -0.19
        f1_ = round(f1(data), 2)
        mcc_ = round(mcc(data), 2)
        self.assertEqual(f1_, 0.17)
        self.assertEqual(mcc_, -0.19)

        # Use Case C2: Negatively imbalanced dataset
        data = {}
        data['ap'] = 11   # 10 positives
        data['an'] = 89   # 89 negatives
        data['se'] = 3
        data['tp'] = 2   # classifier was unable to predict positives
        data['fp'] = 1
        data['tn'] = 88    # classifier able to predict negatives
        data['fn'] = 9
        # Both the scores gives bad measure
        # F1 0.29    MCC 0.31
        f1_ = round(f1(data), 2)
        mcc_ = round(mcc(data), 2)
        self.assertEqual(f1_, 0.29)
        self.assertEqual(mcc_, 0.31)


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
        self.mcc_unittest()

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
