"""Dataset planning with registered fast paths and a black-box refuge."""

from .black_box import BlackBoxPlan, build_black_box_plan
from .registry import Plan, build_plan
from .tensor import TensorPlan, build_tensor_plan

__all__ = [
    "BlackBoxPlan",
    "Plan",
    "TensorPlan",
    "build_black_box_plan",
    "build_plan",
    "build_tensor_plan",
]
