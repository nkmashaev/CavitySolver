import numpy as np
from numpy import linalg as LA


class OutputManager:
    """
    Class OutputManager is designed to output cell centered data
    into Tecplot file
    """

    __slots__ = ("__x", "__y", "__output_dict", "__i_size", "__j_size")

    def __init__(self, x: np.ndarray, y: np.ndarray):
        """
        OutputManager constructor

        :param x: array of x-coordinates
        :param y: array of y-coordinates
        """
        self.__x = x
        self.__y = y
        self.__i_size, self.__j_size = x.shape
        self.__output_dict = {}

    @property
    def header(self) -> str:
        """
        Return names of stored variables in the following format:

        Variables="X", "Y", ...

        :return: stored variable names string
        """
        header = 'Variables="X", "Y"'
        for key in self.__output_dict.keys():
            header += f', "{key}"'
        return header

    @property
    def storage_info(self) -> str:
        """
        Return information about stored arrays dimension, data packing type
        and var location

        :return: storage info string
        """
        param_numb = len(self.__output_dict) + 2
        storage_info = f"Zone i={self.__i_size}, j={self.__j_size}, "
        storage_info += (
            f"DATAPACKING=BLOCK, VARLOCATION=([3-{param_numb}]=CELLCENTERED)"
        )
        return storage_info

    def save_scalar(self, scalar_name: str, scalar_data: np.ndarray):
        """
        Accepts variable name and field and marked it for output.
        Note that variable should be scalar.

        :param scalar_name: name of variable
        :param scalar_data: variable 2D field
        """
        i_size, j_size = scalar_data.shape
        if i_size != self.__i_size + 1 or j_size != self.__j_size + 1:
            raise AttributeError("Error: Dimension mismatch!")
        self.__output_dict[scalar_name] = scalar_data

    def output(self, file_path: str):
        """
        Output marked with save_scalar method data to file in Tecplot format.

        :param file_path: name of file
        """
        with open(file_path, "w") as out_file:
            i = self.__i_size
            j = self.__j_size

            out_file.write(self.header + "\n")
            out_file.write(self.storage_info + "\n")
            np.savetxt(
                out_file, self.__x.T.reshape(1, i * j), fmt="%.11e", delimiter="\n"
            )
            np.savetxt(
                out_file, self.__y.T.reshape(1, i * j), fmt="%.11e", delimiter="\n"
            )
            for val in self.__output_dict.values():
                np.savetxt(
                    out_file,
                    val[1:i, 1:j].T.reshape(1, (i - 1) * (j - 1)),
                    fmt="%.11e",
                    delimiter="\n",
                )
