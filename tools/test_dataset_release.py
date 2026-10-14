import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import prepare_phone_dataset_release as release


class DatasetReleaseTest(unittest.TestCase):
    def make_source(self,base):
        base.mkdir()
        rows=[{'id':str(i),'state':'Synthetic request '+str(i),'questions':json.dumps({'route':{'type':'choice','criteria':{r:r for r in sorted(release.ROUTES)}}}),'gold':json.dumps({'route':{'label':'answer','probabilities':{'answer':1}}}),'reviewer_secret':'excluded'} for i in range(781)]
        train=base/'train.jsonl';train.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        for name in ['val.jsonl','calibration.jsonl','test.jsonl']:(base/name).write_text(json.dumps({'id':name})+'\n')
        return release.sha(train)

    def test_export_train_only_removes_review_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);expected=self.make_source(base/'source')
            manifest=release.prepare(base/'source',expected,base/'output')
            self.assertEqual(manifest['rows'],781)
            self.assertEqual(manifest['published_splits'],['train'])
            self.assertNotIn('reviewer_secret',(base/'output/data/train.jsonl').read_text())
            self.assertFalse((base/'output/data/test.jsonl').exists())

    def test_hash_mismatch_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);self.make_source(base/'source')
            with self.assertRaisesRegex(ValueError,'SHA'):
                release.prepare(base/'source','wrong',base/'output')
            self.assertFalse((base/'output').exists())

    def test_protected_id_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);expected=self.make_source(base/'source')
            (base/'source/test.jsonl').write_text('{"id":"0"}\n')
            with self.assertRaisesRegex(ValueError,'protected'):
                release.prepare(base/'source',expected,base/'output')
            self.assertFalse((base/'output').exists())


if __name__=='__main__':unittest.main()
