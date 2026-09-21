import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from reflexmodels import Choice
from reflexmodels.serialization import DECISION, duplicated_choice_branches


class SerializationTests(unittest.TestCase):
    def test_branches_share_prefix_but_do_not_include_sibling_options(self):
        request = Choice(
            "Kullanıcı config.yaml dosyasını incelemek istiyor.",
            "Hangi araç kullanılmalı?",
            {"filesystem": "Yerel dosyaları oku", "web": "İnternette ara"},
        )
        branches = duplicated_choice_branches(request)
        self.assertIn("Yerel dosyaları oku", branches["filesystem"])
        self.assertNotIn("İnternette ara", branches["filesystem"])
        self.assertTrue(branches["web"].endswith(DECISION))
