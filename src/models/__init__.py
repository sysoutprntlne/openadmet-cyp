"""Model registry for the OpenADMET CYP challenge.

To add a new model: create a module in this directory, subclass CYPModel,
set a unique `name` class variable, and add it here.
"""

from models.base import CYPModel
from models.decision_tree import DecisionTree
from models.ridge import Ridge
from models.TabPFN import TabPFN

REGISTRY: dict[str, type[CYPModel]] = {
    DecisionTree.name: DecisionTree,
    Ridge.name: Ridge,
    TabPFN.name: TabPFN,
}

__all__ = ["CYPModel", "REGISTRY"]
