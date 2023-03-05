import pytest
import numpy as np

from ..utils import utils

class TestGenerate2CornerPoints:
    @staticmethod
    def test_1():
        q = np.array([1])
        with pytest.raises(ValueError):
            utils.generate_2_corner_points(q)

    @staticmethod
    def test_2():
        q = np.array([[5, 5]])
        with pytest.raises(ValueError):
            utils.generate_2_corner_points(q)

    @staticmethod
    def test_3():
        q = np.array([[1], [5]])
        x = np.array([[1], [5]])
        y = np.array([[1], [1]])
        x_test, y_test = utils.generate_2_corner_points(q)
        assert np.all(x == x_test) and np.all(y == y_test)

    @staticmethod
    def test_4():
        q = np.array([[4], [5]])
        x = np.array([[1], [5]])
        y = np.array([[4], [1]])
        x_test, y_test = utils.generate_2_corner_points(q)
        assert np.all(x == x_test) and np.all(y == y_test)

    @staticmethod
    def test_5():
        q = np.array([[5], [5]])
        x = np.array([[5], [1]])
        y = np.array([[1], [5]])
        x_test, y_test = utils.generate_2_corner_points(q)
        assert np.all(x == x_test) and np.all(y == y_test)

    @staticmethod
    def test_6():
        q = np.array([[5], [4], [3], [2], [6]])
        x = np.array([[1], [1], [1], [1], [6]])
        y = np.array([[5], [4], [3], [2], [1]])
        x_test, y_test = utils.generate_2_corner_points(q)
        assert np.all(x == x_test) and np.all(y == y_test)

    @staticmethod
    def test_7():
        q = np.array([[15], [15], [15], [15], [15], [15], [15]])
        x = np.array([[15], [15], [15], [15], [15], [15], [1]])
        y = np.array([[1], [1], [1], [1], [1], [1], [15]])
        x_test, y_test = utils.generate_2_corner_points(q)
        assert np.all(x == x_test) and np.all(y == y_test)

    @staticmethod
    def test_8():
        q = np.array([[15], [18], [11], [3], [22], [64], [1]])
        x = np.array([[1], [1], [1], [1], [1], [64], [1]])
        y = np.array([[15], [18], [11], [3], [22], [1], [1]])
        x_test, y_test = utils.generate_2_corner_points(q)
        assert np.all(x == x_test) and np.all(y == y_test)
