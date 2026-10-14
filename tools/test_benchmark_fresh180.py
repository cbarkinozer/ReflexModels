"""Exercise the complete benchmark without a phone or model inference."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('benchmark', Path(__file__).with_name('benchmark_fresh180.py'))
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class BenchmarkTest(unittest.TestCase):
    def test_complete_run_and_malformed_response(self):
        cases, _, refs = benchmark.load_inputs()
        calls = []

        def adb(command, **kwargs):
            return (benchmark.EXPECTED_MODEL + ' model.gguf\n') if 'sha256sum' in command else 'test telemetry\n'

        def request(req, **kwargs):
            body = json.loads(req.data)
            case = cases[len(calls)]
            self.assertEqual(body['state'], case['state'])
            self.assertEqual(body['questions'], json.loads(case['questions']))
            calls.append(case['id'])
            payload = {'answers': {'route': refs[case['id']]['response']}, 'usage': {'output_tokens': 0}}
            if len(calls) == 2:
                payload['usage']['output_tokens'] = 1
            return contextlib.closing(io.BytesIO(json.dumps(payload).encode()))

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'run'
            argv = ['benchmark', '--out', str(out), '--pid', '123',
                    '--runtime-revision', 'test', '--launch-command', 'test']
            with patch('sys.argv', argv), patch.object(benchmark.subprocess, 'check_output', side_effect=adb), patch.object(benchmark.urllib.request, 'urlopen', side_effect=request), contextlib.redirect_stdout(io.StringIO()):
                benchmark.main()
            summary = json.loads((out / 'summary.json').read_text())
            self.assertEqual(summary['all_requests']['total'], 180)
            self.assertEqual(summary['all_requests']['failures'], 1)
            self.assertEqual(summary['all_requests']['parity_valid_cases'], 179)
            self.assertTrue(json.loads((out / 'manifest.json').read_text())['complete'])
            self.assertEqual(len((out / 'predictions.jsonl').read_text().splitlines()), 180)
            self.assertEqual(len((out / 'telemetry.jsonl').read_text().splitlines()), 180)
            self.assertEqual(summary['all_requests']['reference_agreement'], 1)

    def test_wrong_model_prevents_http(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv = ['benchmark', '--out', str(Path(tmp) / 'run'), '--pid', '123',
                    '--runtime-revision', 'test', '--launch-command', 'test']
            with patch('sys.argv', argv), patch.object(benchmark.subprocess, 'check_output', return_value='wronghash file'), patch.object(benchmark.urllib.request, 'urlopen') as http:
                with self.assertRaisesRegex(AssertionError, 'selected combined'):
                    benchmark.main()
                http.assert_not_called()


if __name__ == '__main__':
    unittest.main()
