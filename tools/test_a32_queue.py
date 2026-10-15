import json
from pathlib import Path
import unittest
from unittest.mock import patch
import a32_queue as q


class QueueScoringTest(unittest.TestCase):
    def test_multi_question_gold_is_explicit_not_distribution_argmax(self):
        rows=[{'gold_labels':{'action':'keep','risk':'1'},'predictions':{'action':'keep','risk':'0'},'latency_ms':10}]
        result=q.summary(rows)
        self.assertEqual(result['decisions'],2)
        self.assertEqual(result['accuracy'],.5)
        self.assertEqual(result['record_accuracy'],0)

    def test_missing_response_counts_all_questions_as_wrong(self):
        result=q.summary([{'gold_labels':{'a':'x','b':'true'},'latency_ms':300000,'error':'timeout'}])
        self.assertEqual(result['accuracy'],0)
        self.assertEqual(result['failures'],1)

    def test_noul_and_score_use_discrete_argmax(self):
        prediction,_=q.question_results({'answers':{'yes':{'noul':.9},'scale':{'score':.8,'probabilities':{'0':.4,'1':.6}}},'usage':{'output_tokens':0}}, {'yes':{'type':'noul'},'scale':{'type':'score','criteria':['low','high']}})
        self.assertEqual(prediction,{'yes':'true','scale':'1'})

    def test_invalid_probability_labels_rejected(self):
        with self.assertRaises(ValueError):
            q.question_results({'answers':{'a':{'choice':'x','probabilities':{'x':1}}},'usage':{'output_tokens':0}}, {'a':{'type':'choice','criteria':{'x':'x','y':'y'}}})

    def test_grammar_preserves_escaped_labels(self):
        grammar=q.grammar({'q':{'type':'choice','criteria':{'quote"label':'x','back\\slash':'y'}}})
        self.assertIn(json.dumps(json.dumps('quote"label')),grammar)
        self.assertIn(json.dumps(json.dumps('back\\slash')),grammar)

    def test_cache_profiles_change_only_the_cache_flag(self):
        questions={'route':{'type':'choice','criteria':{'answer':'ordinary','get_battery_info':'battery'}}}
        for profile in ['raw','legacy']:
            off=q.baseline_body({'state':'A fixture input'},questions,profile)
            on=q.baseline_body({'state':'A fixture input'},questions,profile+'-cached')
            self.assertFalse(off['cache_prompt'])
            self.assertEqual(off|{'cache_prompt':True},on)
            self.assertEqual(off['max_tokens'],64 if profile=='legacy' else 256)

    def test_existing_router_action_normalization(self):
        questions={'route':{'type':'choice','criteria':{'answer':'a','get_battery_info':'b','phone_health_check':'c'}}}
        fixtures=[({'action':'answer','text':''},'answer'),({'action':'tool','name':'get_battery_info','args':{}},'get_battery_info'),({'action':'workflow','name':'phone_health_check','args':{}},'phone_health_check')]
        for payload,label in fixtures:
            response={'choices':[{'text':json.dumps(payload)}]}
            self.assertEqual(q.baseline_predictions(response,questions,'legacy'),{'route':label})
        with self.assertRaisesRegex(ValueError,'workflow'):
            q.baseline_predictions({'choices':[{'text':'{"action":"workflow","name":"get_battery_info"}'}]},questions,'legacy')

    def test_all_frozen_general_cohorts_validate(self):
        directory=q.BASE/'heldout-results-20261014/frozen-general-tests'
        manifest=json.loads((directory/'manifest.json').read_text())
        for name,n in [('typed_decisions',400),('banking77',462),('clinc150',450)]:
            path=directory/(name+'.jsonl')
            self.assertEqual(q.sha(path),manifest['files'][name]['sha256'])
            rows=[json.loads(l) for l in path.read_text().splitlines()]
            self.assertEqual(len(rows),n)
            for r in rows:
                questions=q.decode(r['questions']);gold=q.decode(r['gold'])
                self.assertEqual(set(gold),set(questions))
                for key,question in questions.items():self.assertIn(str(gold[key]['label']),q.options(question))


if __name__=='__main__':unittest.main()
