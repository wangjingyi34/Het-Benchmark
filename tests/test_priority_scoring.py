import unittest

from src.priority import OperatorPriorityEvidence, OperatorPriorityScorer

try:
    import numpy as np
    import torch  # noqa: F401
    from src.kg_a2o import KGA2O
    HAS_TORCH_RUNTIME = True
except ImportError:
    HAS_TORCH_RUNTIME = False


class OperatorPriorityScoringTest(unittest.TestCase):
    def test_paper_equation(self):
        evidence = OperatorPriorityEvidence(
            operator_id="attention_0",
            attribution=0.8,
            headroom=0.5,
            uncertainty=0.1,
            implementation_risk=0.2,
        )
        self.assertAlmostEqual(evidence.score, 0.8 * 0.5 * 0.9 * 0.8)

    def test_full_uncertainty_or_risk_suppresses_priority(self):
        uncertain = OperatorPriorityEvidence("op_u", 1.0, 1.0, 1.0, 0.0)
        risky = OperatorPriorityEvidence("op_r", 1.0, 1.0, 0.0, 1.0)
        self.assertEqual(uncertain.score, 0.0)
        self.assertEqual(risky.score, 0.0)

    def test_invalid_or_missing_components_are_rejected(self):
        with self.assertRaises(ValueError):
            OperatorPriorityEvidence("op", 1.1, 0.5, 0.1, 0.1)
        with self.assertRaises(KeyError):
            OperatorPriorityScorer.rank([
                {
                    "operator_id": "op",
                    "attribution": 0.5,
                    "headroom": 0.5,
                    "uncertainty": 0.1,
                }
            ])

    def test_ranking_is_descending_and_stable_for_ties(self):
        ranked = OperatorPriorityScorer.rank([
            {
                "operator_id": "low",
                "attribution": 0.2,
                "headroom": 0.5,
                "uncertainty": 0.0,
                "implementation_risk": 0.0,
            },
            {
                "operator_id": "high",
                "attribution": 0.8,
                "headroom": 0.5,
                "uncertainty": 0.0,
                "implementation_risk": 0.0,
            },
            {
                "operator_id": "high_tie",
                "attribution": 0.8,
                "headroom": 0.5,
                "uncertainty": 0.0,
                "implementation_risk": 0.0,
            },
        ])
        self.assertEqual([item["operator_id"] for item in ranked], [
            "high", "high_tie", "low"
        ])

    @unittest.skipUnless(HAS_TORCH_RUNTIME, "PyTorch runtime is not installed")
    def test_optimize_consumes_priority_order_without_mutating_input(self):
        operators = [
            {"id": "low", "type": "MatMul", "latency": 1.0, "embedding": np.zeros(64)},
            {"id": "high", "type": "MatMul", "latency": 1.0, "embedding": np.zeros(64)},
        ]
        evidence = [
            {
                "operator_id": "low",
                "attribution": 0.1,
                "headroom": 0.5,
                "uncertainty": 0.0,
                "implementation_risk": 0.0,
            },
            {
                "operator_id": "high",
                "attribution": 0.9,
                "headroom": 0.5,
                "uncertainty": 0.0,
                "implementation_risk": 0.0,
            },
        ]
        optimizer = KGA2O(device="cpu")
        plan = optimizer.optimize(
            operators,
            np.zeros(64),
            priority_evidence=evidence,
        )
        self.assertEqual([operator_id for operator_id, _ in plan], ["high", "low"])
        self.assertEqual([operator["id"] for operator in operators], ["low", "high"])


if __name__ == "__main__":
    unittest.main()
