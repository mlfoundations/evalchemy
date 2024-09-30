import logging
import os
from typing import List, Optional

import ray

from datasets import Dataset, concatenate_datasets
from engine.dag import DAG
from engine.operators.operator import ManyShardRefs

logger = logging.getLogger(__name__)

class DAGExecutor:
    def __init__(self, dag: DAG):
        self.dag = dag

    def get_waitables(self) -> ManyShardRefs:
        """
        Execute the operators in the DAG and return a promise of the list of shards at the end of the data generation process.

        Returns:
            ManyShardRefs: References to the output shards of the data generation process.
        """
        datas = {}
        sorted_operators = self.dag.topological_sort()
        waitables = []

        for operator in sorted_operators:
            input_datas = {input_id: datas[input_id] for input_id in operator.input_ids}
            curr_op_output = operator.execute(input_datas)
            datas[operator.id] = curr_op_output
            if operator.id in self.dag.output_ids:
                waitables.extend(curr_op_output)

        logger.info(f"Generated {len(waitables)} waitables")
        return waitables

    def run(self) -> Dataset:
        """
        Run the entire data generation process.

        This method initializes Ray, executes the DAG, and processes the results.
        """
        waitables = self.get_waitables()
        
        if not waitables:
            logger.error("No waitables generated. Check your DAG configuration.")
            return

        logger.info(f"Waiting for {len(waitables)} tasks to complete")
        ray.wait(waitables, num_returns=len(waitables))
        
        logger.info(f"Done with execution.")
        try:
            results = [ray.get(shard) for shard in waitables]
            logger.info(f"Retrieved {len(results)} results")

            non_empty_results = [r for r in results if r is not None and len(r) > 0]
            if not non_empty_results:
                logger.error("All retrieved results are empty or None")
                ray.shutdown()
                return

            results = concatenate_datasets(non_empty_results)
            logger.info(f"Concatenated dataset has {len(results)} rows")
            return results
        except Exception as e:
            logger.error(f"Error during result processing: {str(e)}")
        finally:
            ray.shutdown()
