import os
import unittest
from collections import Counter

import ray

from datasets import Dataset, concatenate_datasets
from engine.dag import load_dag
from engine.executor import DAGExecutor
from engine.operators.load_preexisting_operator import (
    LoadPreexistingOperator,
    LoadPreexistingOperatorConfig,
)
from engine.operators.mix_operator import MixOperatorConfig
from engine.operators.operator import OperatorConfig, create_operator
from engine.tests.dummy_source_operator import register_dummy_operator


class TestOperators(unittest.TestCase):
    def setUp(self):
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        register_dummy_operator()
        self.test_strategies_dir = os.path.dirname(__file__)
        self.test_configs_dir = os.path.join(self.test_strategies_dir, "test_configs")

    @classmethod
    def tearDownClass(cls):
        ray.shutdown()

    def test_load_preexisting_operator(self):
        config = LoadPreexistingOperatorConfig(
            type="load_preexisting",
            framework_name="simple_test_strategy",
            strategies_dir=self.test_strategies_dir
        )
        operator = LoadPreexistingOperator(id="test_load", input_ids=[], config=config)
        
        result = operator.execute({})
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)  # Two shards from dummy_source
        
        # Get the actual datasets
        datasets = ray.get(result)
        combined_dataset = concatenate_datasets(datasets)
        
        self.assertEqual(len(combined_dataset), 10)  # 5 rows * 2 shards
        self.assertIn("id", combined_dataset.column_names)
        self.assertIn("output", combined_dataset.column_names)
        
        # Check if dummy_uppercase function was applied
        for item in combined_dataset:
            self.assertTrue(item['output'].isupper(), f"Input not uppercase: {item['output']}")

    def test_mix_operator_with_dummy_source(self):
        # Load the DAG from the YAML file
        dag = load_dag(os.path.join(self.test_configs_dir, "dummy_mix_test.yaml"))
        
        # Create and run the DAGExecutor
        executor = DAGExecutor(dag)
        result = executor.run()
        
        self.assertIsInstance(result, Dataset)
        self.assertEqual(len(result), 12)  # 2 sources * 2 shards * 3 rows
        
        # Check if the data points are the same as expected
        expected_outputs = set(f"Sample text {i}" for i in range(3)) | set(f"Sample text {i}" for i in range(3))
        actual_outputs = set(result["output"])
        self.assertEqual(expected_outputs, actual_outputs, "Data points are not as expected after mixing")
        
        # Check if the order is different from the original
        original_order = [f"Sample text {i}" for i in range(3)] * 2 + [f"Sample text {i}" for i in range(3)] * 2
        mixed_order = list(result["output"])
        self.assertNotEqual(original_order, mixed_order, "Order of data points is not shuffled")

if __name__ == "__main__":
    unittest.main()