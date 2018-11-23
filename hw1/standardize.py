import numpy as np


class Standardize(object):
    __mean = None
    __std = None

    def standardize(self, data):
        new = np.array(data)
        self.__mean = np.mean(new, axis=0)
        self.__std = np.std(new, axis=0)
        new = (new - self.__mean) / self.__std
        return new

    def destandardize(self, data):
        new = np.array(data)
        new = new * self.__std + self.__mean
        return new
