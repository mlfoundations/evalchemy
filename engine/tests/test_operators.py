import os
import unittest

import ray

from datasets import Dataset
from engine.operators.load_preexisting_operator import (
    LoadPreexistingOperator,
    LoadPreexistingOperatorConfig,
)
from engine.operators.mix_operator import MixOperator, MixOperatorConfig


class TestOperators(unittest.TestCase):
    def setUp(self):
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        self.sample_dataset = Dataset.from_dict({"text": ["Hello", "World", "Test", "Data"]})
        self.test_strategies_dir = os.path.dirname(__file__)

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
        self.assertEqual(len(result), 1)
        self.assertTrue(isinstance(result[0], ray.ObjectRef))
        
        # Get the actual dataset
        dataset = ray.get(result)
        print(dataset)
        self.assertIsInstance(dataset, Dataset)
        self.assertEqual(list(dataset["text"]), ["Dummy", "Data", "For", "Testing"])

    def test_mix_operator(self):
        for num_shards in [1, 2, 3]:
            with self.subTest(num_shards=num_shards):
                config = MixOperatorConfig(type="mix", seed=42)
                operator = MixOperator(id="test_mix", input_ids=["input1", "input2"], config=config)
                
                # Create input shards
                input_shards = [ray.put(self.sample_dataset.select(range(i, i+2))) for i in range(0, len(self.sample_dataset), 2)]
                inputs = {"input1": input_shards[:num_shards], "input2": input_shards[num_shards:]}
                
                result = operator.execute(inputs)
                self.assertIsInstance(result, list)
                self.assertEqual(len(result), 1) 
                
                # Get the mixed dataset
                mixed_dataset = ray.get(result[0])
                self.assertIsInstance(mixed_dataset, Dataset)
                self.assertEqual(len(mixed_dataset), len(self.sample_dataset))
                
                # Check if the dataset is shuffled (order should be different from the original but same elements)
                self.assertNotEqual(list(mixed_dataset["text"]), list(self.sample_dataset["text"]))
                self.assertEqual(set(mixed_dataset["text"]), set(self.sample_dataset["text"]))

if __name__ == "__main__":
    unittest.main()