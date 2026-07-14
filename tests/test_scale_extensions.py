import json
from pathlib import Path
import unittest


class ScaleExtensionMetadataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).parents[1] / "data" / "model_scale_extensions.json"
        cls.payload = json.loads(path.read_text(encoding="utf-8"))

    def test_extensions_are_not_mixed_into_profiled_counts(self):
        metadata = self.payload["metadata"]
        self.assertFalse(metadata["included_in_profiled_core"])
        self.assertEqual(metadata["profiled_core_models"], 34)
        self.assertEqual(metadata["profiled_core_operators"], 6244)

    def test_official_scale_records_are_present(self):
        by_name = {model["name"]: model for model in self.payload["models"]}
        self.assertEqual(by_name["Llama-3.1-405B"]["parameters"], 405_000_000_000)
        self.assertEqual(by_name["Qwen2.5-72B"]["parameters"], 72_000_000_000)
        self.assertEqual(
            by_name["Stable-Diffusion-3-Suite"]["parameter_range"],
            {"min": 800_000_000, "max": 8_000_000_000},
        )

    def test_every_record_has_a_primary_source(self):
        for model in self.payload["models"]:
            self.assertTrue(model["source"].startswith("https://"))
            self.assertTrue(model["source_fact"])


if __name__ == "__main__":
    unittest.main()
