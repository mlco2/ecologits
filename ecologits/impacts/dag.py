from functools import wraps
from graphlib import TopologicalSorter
from typing import Any, Callable


class DAG:
    """
    Directed Acyclic Graph (DAG) of computations
    """

    def __init__(self) -> None:
        self.__tasks: dict[str, Callable] = {}
        self.__dependencies: dict[str, set] = {}

    def asset(self, func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

        # Register the task and its dependencies
        self.__tasks[func.__name__] = func
        func_params = list(func.__annotations__.keys())[:-1]  # Ignore return type
        self.__dependencies[func.__name__] = set(func_params)

        return wrapper

    def build_dag(self) -> TopologicalSorter:
        return TopologicalSorter(self.__dependencies)

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        ts = self.build_dag()
        results = kwargs.copy()  # Use initial params as the starting point

        for task_name in ts.static_order():
            if task_name in results:  # Skip execution if result already provided
                continue
            task = self.__tasks.get(task_name)
            if task is None:  # Leaf input that was not provided (only valid when nothing executed reads it)
                continue
            # Collect results from dependencies or use initial params
            dep_results = {dep: results.get(dep) for dep in self.__dependencies[task_name]}
            # Filter out None values if not all dependencies are met
            dep_results = {k: v for k, v in dep_results.items() if v is not None}
            results[task_name] = task(**dep_results)

        return results
