import os
from typing import Tuple


def is_integer(str_to_check: str) -> bool:
    try:
        int(str_to_check)
    except ValueError:
        return False
    else:
        return True


def is_float(str_to_check: str) -> bool:
    try:
        float(str_to_check)
    except ValueError:
        return False
    else:
        return True


class InputManager:
    """
    Class InputManager is designed to parse init_file data
    to program
    """

    __slots__ = "__storage"

    def __init__(self, file_path: str):
        self.__storage = {}
        with open(file_path, "r") as in_file:
            for line in in_file:
                line = line.strip().split("#", 1)[0]
                try:
                    key, val, *_ = line.strip().split("=", 1)
                except ValueError:
                    continue

                if not key.isidentifier():
                    raise ValueError(
                        f"Expected: Unacceptable attribute name({key}) is given!"
                    )
                if is_integer(val):
                    val = int(val)
                elif is_float(val):
                    val = float(val)
                self.__storage[key] = val
        self.__check_init()

    def __check_init(self):
        if not "mesh" in self.__storage:
            raise AttributeError("Error: Mesh file name not found!")
        if not isinstance(self.__storage["mesh"], str):
            raise TypeError("Error: Mesh file name expected to be string!")

        if not "Re" in self.__storage:
            raise AttributeError("Error: Re parameter is not defined!")
        if not isinstance(self.__storage["Re"], (int, float)):
            raise TypeError("Error: Re number expected to be float!")
        if self.__storage["Re"] <= 0:
            raise ValueError("Error: Re number expected to be positive!")

        if not "CFL" in self.__storage:
            raise AttributeError("Error: CFL parameter is not defined!")
        if not isinstance(self.__storage["CFL"], (int, float)):
            raise TypeError("Error: CFL number expected to be float!")
        if self.__storage["CFL"] <= 0:
            raise ValueError("Error: CFL number expected to be positive!")

        if not "ACP" in self.__storage:
            raise AttributeError("Error: ACP parameter is not defined!")
        if not isinstance(self.__storage["ACP"], (int, float)):
            raise TypeError(
                "Error: Artificial Compressibility parameter" + " expected to be float!"
            )
        if self.__storage["ACP"] <= 0:
            raise ValueError(
                "Error: Artificial compressibility parameter"
                + " expected to be positive!"
            )

        if not isinstance(self.__storage.get("loctime", 0), int):
            raise TypeError("Error: Local time method id expected to be integer!")
        if not (0 <= self.__storage.get("loctime", 0) <= 1):
            raise ValueError("Error: Unknowm local time method id!")

        if not isinstance(self.__storage.get("smoothing", 0), int):
            raise TypeError("Error: Smoothing method id expected to be integer!")
        if not (0 <= self.__storage.get("smoothing", 0) <= 1):
            raise ValueError("Error: Unknown smoothing method id!")

        if not isinstance(self.__storage.get("RK", 1), int):
            raise TypeError(
                "Error: Runge-Kutta number of levels expected to be integer!"
            )
        if not (1 <= self.__storage.get("RK", 1) <= 5):
            raise ValueError("Error: Unknown RK level coefficients!")

        if not isinstance(self.__storage.get("grad", 0), int):
            raise TypeError(
                "Error: Gradient calculation approach id" + "expected to be integer!"
            )
        if not (0 <= self.__storage.get("grad", 0) <= 1):
            raise ValueError("Error: Unknown gradient calculation approach id!")

        if not isinstance(self.__storage.get("gauss_iter", 1), int):
            raise TypeError(
                "Error: Number of green gauss method's"
                + "iteration expected to be integer!"
            )
        if not self.__storage.get("gauss_iter", 1) > 0:
            raise ValueError(
                "Error: Number of green gauss method's"
                + "iteration expected to be positive!"
            )

    @property
    def msh(self) -> str:
        return self.__storage["mesh"]

    @property
    def grad(self) -> Tuple[int, str]:
        grad_scheme = self.__storage.get("grad", 0)
        if grad_scheme == 0:
            return grad_scheme, "Green Gauss"
        if grad_scheme == 1:
            return grad_scheme, "Least Squares"

    @property
    def gauss_iter(self) -> int:
        return self.__storage.get("gauss_iter", 1)

    @property
    def outfile(self) -> str:
        return self.__storage.get("outfile", "data.plt")

    @property
    def Re(self) -> float:
        return self.__storage["Re"]

    @property
    def CFL(self) -> float:
        return self.__storage["CFL"]

    @property
    def ACP(self) -> float:
        return self.__storage["ACP"]

    @property
    def loctime(self) -> float:
        return self.__storage.get("loctime", 0)

    @property
    def smoothing(self) -> float:
        return self.__storage.get("smoothing", 0)

    @property
    def RK(self) -> int:
        return self.__storage.get("RK", 1)
