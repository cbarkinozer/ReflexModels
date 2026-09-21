import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from reflexmodels import Binary, Choice, Ordinal


class DecisionTypeTests(unittest.TestCase):
    def test_choice_accepts_runtime_defined_turkish_options(self):
        request = Choice(
            state="Kullanıcı config.yaml dosyasını incelemek istiyor.",
            question="Hangi araç kullanılmalı?",
            options={"filesystem": "Yerel dosyaları oku", "web": "İnternette ara"},
        )
        self.assertEqual(request.language, "tr")
        self.assertEqual(request.options["filesystem"], "Yerel dosyaları oku")
        with self.assertRaises(TypeError):
            request.options["shell"] = "Komut çalıştır"

    def test_choice_requires_multiple_options(self):
        with self.assertRaisesRegex(ValueError, "at least two"):
            Choice("durum", "soru", {"only": "tek seçenek"})

    def test_ordinal_preserves_order_and_rejects_duplicates(self):
        request = Ordinal("yanıt", "Risk düzeyi nedir?", ("düşük", "orta", "yüksek"))
        self.assertEqual(request.levels[2], "yüksek")
        with self.assertRaisesRegex(ValueError, "unique"):
            Ordinal("yanıt", "Risk düzeyi nedir?", ("düşük", "düşük"))

    def test_binary_requires_a_question(self):
        with self.assertRaisesRegex(ValueError, "question"):
            Binary("Sistem çalışıyor.", "")


if __name__ == "__main__":
    unittest.main()
