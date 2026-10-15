import json
import math
from pathlib import Path
import tempfile
import unittest
import report_a32 as report


class ReportTest(unittest.TestCase):
    def write(self,directory,records):
        directory.mkdir()
        (directory/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))

    def test_calibration_uses_selected_probability_and_hard_gold(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)/'run'
            self.write(directory,[{'id':'a','gold_labels':{'q':'true'},'predictions':{'q':'true'},'probabilities':{'q':{'true':.8,'false':.2}}}])
            result=report.calibration(directory)
            self.assertAlmostEqual(result['ece_10_equal_width_bins'],.2)
            self.assertAlmostEqual(result['mean_multiclass_brier_sum'],.08)
            self.assertAlmostEqual(result['hard_label_nll'],-math.log(.8))

    def test_cache_statistics_use_reported_reuse_not_requested_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)/'run'
            records=[{'request_body':{'cache_prompt':True},'raw_response':json.dumps({'usage':{'prompt_tokens':100,'completion_tokens':10,'prompt_tokens_details':{'cached_tokens':c}}})} for c in [0,75]]
            self.write(directory,records)
            result=report.usage_statistics(directory)
            self.assertEqual(result['requests_reporting_nonzero_cached_tokens'],1)
            self.assertEqual(result['cached_fraction_of_reported_input_tokens'],.375)
            self.assertEqual(result['requested_cache_values'],[True])

    def test_paired_difference_requires_same_state_and_gold(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=Path(tmp)/'a';b=Path(tmp)/'b'
            left=[{'id':'one','state':'same','gold_labels':{'route':'x'},'predictions':{'route':'x'}}]
            right=[{'id':'one','state':'different','gold_labels':{'route':'x'},'predictions':{'route':'x'}}]
            self.write(a,left);self.write(b,right)
            with self.assertRaisesRegex(ValueError,'Unmatched states'):report.paired(a,b)

    def test_paired_improvement_counts_discordant_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=Path(tmp)/'a';b=Path(tmp)/'b'
            left=[{'id':str(i),'state':'same','gold_labels':{'route':'x'},'predictions':{'route':'x'}} for i in range(2)]
            right=[r|{'predictions':{'route':'y'}} for r in left]
            self.write(a,left);self.write(b,right)
            result=report.paired(a,b)
            self.assertEqual(result['reflex_only_correct'],2)
            self.assertEqual(result['baseline_only_correct'],0)
            self.assertEqual(result['accuracy_difference'],1)
            self.assertEqual(result['paired_bootstrap_95ci'],[1,1])
            self.assertEqual(result['exact_mcnemar_two_sided_p'],.5)


if __name__=='__main__':unittest.main()
