"""Module containing treatment planning problem definitions."""

from ._optiprob import NonLinearPlanningProblem, PlanningProblem
from ._nonlin_fluence import NonLinearFluencePlanningProblem
from ._simple_least_squares import SimpleLeastSquaresFluenceOptimization
from ._factory import get_available_problems, get_problem, get_problem_from_pln, register_problem

register_problem(SimpleLeastSquaresFluenceOptimization)
register_problem(NonLinearFluencePlanningProblem)

__all__ = [
    "SimpleLeastSquaresFluenceOptimization",
    "NonLinearFluencePlanningProblem",
    "NonLinearPlanningProblem",
    "PlanningProblem",
    "get_available_problems",
    "get_problem",
    "get_problem_from_pln",
]
