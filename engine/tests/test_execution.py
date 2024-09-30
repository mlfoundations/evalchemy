import logging
import unittest

import ray

from engine.dag import parse_dag
from engine.executor import DAGExecutor
from engine.tests.dummy_source_operator import register_dummy_operator

logger = logging.getLogger(__name__)

class TestExecution(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ray.init()
        logging.basicConfig(level=logging.INFO)
        register_dummy_operator()

    @classmethod
    def tearDownClass(cls):
        ray.shutdown()

    def test_execution_with_dummy_source(self):
        config = {
            "name": "test_execution",
            "operators": [
                {
                    "id": "source",
                    "config": {
                        "type": "dummy_source",
                        "num_rows": 5
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
        result = executor.run()

        self.assertIsNotNone(result, "Generated dataset is None")
        if result is not None:
            self.assertEqual(len(result), 10, f"Expected 10 rows (5 rows * 2 shards), but got {len(result)}")
            
            # Check if the transformations were applied correctly
            for item in result:
                self.assertTrue('output' in item, f"'output' key not found in item: {item}")
                self.assertTrue(item['output'].isupper(), f"Input not uppercase: {item['output']}")
                self.assertTrue(item['output'].endswith('!'), f"Input doesn't end with '!': {item['output']}")

if __name__ == '__main__':
    unittest.main()