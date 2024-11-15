from engine.operators.completions_operator import (
    CompletionsOperator,
    CompletionsOperatorConfig,
)
from engine.operators.dclm_refinedweb_source_operator import (
    DCLMRefineWebSourceConfig,
    DCLMRefineWebSourceOperator,
)
from engine.operators.fasttext_operator import FastTextOperator, FastTextOperatorConfig
from engine.operators.function_operator import FunctionOperator, FunctionOperatorConfig
from engine.operators.hf_source_operator import HFSourceOperator, HFSourceOperatorConfig
from engine.operators.mix_operator import MixOperator, MixOperatorConfig
from engine.operators.operator import register_operator
from engine.operators.remove_columns_operator import (
    RemoveColumnsOperator,
    RemoveColumnsOperatorConfig,
)
from engine.operators.rename_column_operator import (
    RenameColumnOperator,
    RenameColumnOperatorConfig,
)

register_operator(FunctionOperatorConfig, FunctionOperator)
register_operator(MixOperatorConfig, MixOperator)
register_operator(HFSourceOperatorConfig, HFSourceOperator)
register_operator(RenameColumnOperatorConfig, RenameColumnOperator)
register_operator(RemoveColumnsOperatorConfig, RemoveColumnsOperator)
register_operator(FastTextOperatorConfig, FastTextOperator)
register_operator(DCLMRefineWebSourceConfig, DCLMRefineWebSourceOperator)
register_operator(CompletionsOperatorConfig, CompletionsOperator)

__all__ = [
    "FunctionOperator",
    "FunctionOperatorConfig",
    "LoadPreexistingOperator",
    "LoadPreexistingOperatorConfig",
    "MixOperator",
    "MixOperatorConfig",
    "HFSourceOperator",
    "HFSourceOperatorConfig",
    "RenameColumnOperator",
    "RenameColumnOperatorConfig",
    "RemoveColumnsOperator",
    "RemoveColumnsOperatorConfig",
    "DAGOperator",
    "DAGOperatorConfig",
    "CompletionsOperator",
]
