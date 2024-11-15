import unittest

from pydantic import ValidationError

from engine.dag import parse_dag
from engine.operators.operator import OperatorSpecificConfig


class TestDAG(unittest.TestCase):
    def test_parse_dag_with_defaults(self):
        config = {
            "name": "test_dag",
            "operators": [
                {"id": "op1", "config": {"type": "hf_source", "dataset": "dataset1", "split": "train"}},
                {
                    "id": "op2",
                    "config": {"type": "function", "function": "engine.tests.dummy_functions.dummy_function1"},
                },
                {
                    "id": "op3",
                    "config": {"type": "function", "function": "engine.tests.dummy_functions.dummy_function2"},
                },
            ],
        }

        dag = parse_dag(config)

        self.assertEqual(dag.name, "test_dag")
        self.assertEqual(len(dag.operators), 3)

        # Check default input_ids
        self.assertEqual(dag.operators[0].input_ids, [])
        self.assertEqual(dag.operators[1].input_ids, ["op1"])
        self.assertEqual(dag.operators[2].input_ids, ["op2"])

        # Check default output_ids
        self.assertEqual(dag.output_ids, ["op3"])

    def test_parse_dag_with_explicit_inputs_and_outputs(self):
        config = {
            "name": "test_dag_explicit",
            "operators": [
                {"id": "op1", "config": {"type": "hf_source", "dataset": "dataset1", "split": "train"}},
                {
                    "id": "op2",
                    "input_ids": ["op1"],
                    "config": {"type": "function", "function": "engine.tests.dummy_functions.dummy_function1"},
                },
                {
                    "id": "op3",
                    "input_ids": ["op1", "op2"],
                    "config": {"type": "function", "function": "engine.tests.dummy_functions.dummy_function2"},
                },
            ],
            "output_ids": ["op2", "op3"],
        }

        dag = parse_dag(config)

        self.assertEqual(dag.name, "test_dag_explicit")
        self.assertEqual(len(dag.operators), 3)

        # Check explicit input_ids
        self.assertEqual(dag.operators[0].input_ids, [])
        self.assertEqual(dag.operators[1].input_ids, ["op1"])
        self.assertEqual(dag.operators[2].input_ids, ["op1", "op2"])

        # Check explicit output_ids
        self.assertEqual(dag.output_ids, ["op2", "op3"])

    def test_parse_dag_mixed_defaults_and_explicit(self):
        config = {
            "name": "test_dag_mixed",
            "operators": [
                {"id": "op1", "config": {"type": "hf_source", "dataset": "dataset1", "split": "train"}},
                {
                    "id": "op2",
                    "input_ids": ["op1"],
                    "config": {"type": "function", "function": "engine.tests.dummy_functions.dummy_function1"},
                },
                {
                    "id": "op3",
                    "config": {"type": "function", "function": "engine.tests.dummy_functions.dummy_function2"},
                },
            ],
        }

        dag = parse_dag(config)

        self.assertEqual(dag.name, "test_dag_mixed")
        self.assertEqual(len(dag.operators), 3)

        # Check mixed input_ids
        self.assertEqual(dag.operators[0].input_ids, [])
        self.assertEqual(dag.operators[1].input_ids, ["op1"])
        self.assertEqual(dag.operators[2].input_ids, ["op2"])

        # Check default output_ids
        self.assertEqual(dag.output_ids, ["op3"])


if __name__ == "__main__":
    unittest.main()
