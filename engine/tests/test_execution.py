import logging
import unittest

import ray

from engine.dag import parse_dag
from engine.executor import DAGExecutor

logger = logging.getLogger(__name__)

class TestExecution(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ray.init()
        logging.basicConfig(level=logging.INFO)

    @classmethod
    def tearDownClass(cls):
        ray.shutdown()

    def test_execution_with_hf_source(self):
        config = {
            "name": "test_execution",
            "operators": [
                {
                    "id": "source",
                    "config": {
                        "type": "hf_source",
                        "dataset": "yahma/alpaca-cleaned",
                        "split": "train",
                        "num_truncate": 5
                    }
                },
                {
                    "id": "uppercase",
                    "config": {
                        "type": "function",
                        "function": "engine.tests.dummy_functions.dummy_uppercase"
                    }
                },
                {
                    "id": "add_exclamation",
                    "config": {
                        "type": "function",
                        "function": "engine.tests.dummy_functions.dummy_add_exclamation"
                    }
                }
            ],
            "output_ids": ["add_exclamation"]
        }
        dag = parse_dag(config)
        executor = DAGExecutor(dag)
        executor.run()

        result = executor.get_generated_dataset()

        self.assertIsNotNone(result, "Generated dataset is None")
        if result is not None:
            self.assertEqual(len(result), 5, f"Expected 5 rows, but got {len(result)}")
            
            # Check if the transformations were applied correctly
            for item in result:
                self.assertTrue('output' in item, f"'output' key not found in item: {item}")
                self.assertTrue(item['output'].isupper(), f"Input not uppercase: {item['output']}")
                self.assertTrue(item['output'].endswith('!'), f"Input doesn't end with '!': {item['output']}")

if __name__ == '__main__':
    unittest.main()