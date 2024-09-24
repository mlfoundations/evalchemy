from typing import Dict, List, TypeAlias

import ray

DatasetRefShard: TypeAlias = ray.ObjectRef
DatasetRef: TypeAlias = List[DatasetRefShard]
DatasetRefs: TypeAlias = Dict[str, DatasetRef]
