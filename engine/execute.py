import logging
from typing import Any, Dict

import ray

from engine.parser import parse_dag


def execute_pipeline(config_path: str):
    # Initialize Ray
    ray.init()

    try:
        # Parse the YAML into a linearized list of operators
        linearized_dag = parse_dag(config_path)

        # Main execution loop
        datas = {}
        for operator in linearized_dag:
            input_datas = {input_id: datas[input_id] for input_id in operator.input_ids}
            datas[operator.id] = operator.execute(input_datas)

        # Wait for all tasks to complete and retrieve results
        waitables = [data_shard for data in datas.values() for data_shard in data]
        ray.wait(waitables, num_returns=len(waitables))
        logging.info("Execution completed. Results.")

    finally:
        # Shut down Ray
        ray.shutdown()


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python -m engine.execute <config_path>")
        sys.exit(1)

    config_path = sys.argv[1]
    execute_pipeline(config_path)
