from typing import Dict, Type
from engine.operators.operator import Operator
from engine.operators.configs import OperatorSpecificConfig

OPERATOR_MAP: Dict[Type[OperatorSpecificConfig], Type[Operator]] = {}
CONFIG_TYPE_MAP: Dict[str, Type[OperatorSpecificConfig]] = {}

def register_operator(config_class: Type[OperatorSpecificConfig], operator_class: Type[Operator]):
    OPERATOR_MAP[config_class] = operator_class
    CONFIG_TYPE_MAP[config_class.model_fields["type"].default] = config_class


def get_operator_class(config_class: Type[OperatorSpecificConfig]) -> Type[Operator]:
    return OPERATOR_MAP.get(config_class)


def get_config_class(config_type: str) -> Type[OperatorSpecificConfig]:
    return CONFIG_TYPE_MAP.get(config_type)
