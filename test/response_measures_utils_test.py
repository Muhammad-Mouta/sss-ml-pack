import pytest
import numpy as np

# Import from parent directory
from ..utils.response_measures import ODResponseMeasure as odu
from ..utils.response_measures import MultimodalDistroModel as mdmu
from ..utils.utils import compute_cosine_max_distance_brute_force


class TestComputeM:
    @staticmethod
    def test_1():
        R = np.array([[1]])
        assert np.all(mdmu.compute_m(R) == np.array([[1]]))

    @staticmethod
    def test_2():
        R = np.array([[1], [1]])
        assert np.all(mdmu.compute_m(R) == np.array([[1], [1]]))

    @staticmethod
    def test_3():
        R = np.array([[1, 1]])
        assert np.all(mdmu.compute_m(R) == np.array([[2]]))

    @staticmethod
    def test_4():
        R = np.array([[1, 1, 1, 0],
                      [1, 1, 0, 0],
                      [1, 1, 1, 1]])
        assert np.all(mdmu.compute_m(R) == np.array([[3],
                                              [2],
                                              [4]]))


class TestComputeS:
    @staticmethod
    def test_1():
        s = None
        R = np.array([[1]], dtype="float64")
        assert np.all(mdmu.compute_s(s, R) == 1/4)

    @staticmethod
    def test_2():
        s = None
        R = np.array([[1, 0, 0],
                      [1, 1, 0],
                      [1, 1, 1]], dtype="float64")
        assert np.all(mdmu.compute_s(s, R) == np.array([[1/4],
                                                        [2/4],
                                                        [3/4]]))

    @staticmethod
    def test_3():
        s = np.array([[0.5], [1.25], [0], [None]], dtype="float64")
        R = np.array([[5, 5, 0, 0, 0, 0, 0, 0, 0, 0],
                      [1, 2, 4, 5, 0, 0, 0, 0, 0, 0],
                      [1, 1, 1, 1, 1, 1, 1, 1, 9, 1],
                      [5, 1, 1, 5, 1, 1, 0, 0, 0, 0]], dtype="float64")
        assert np.all(mdmu.compute_s(s, R) == np.array([[0.5], [1.25], [0], [6/4]], dtype="float64"))


class TestComputeV:
    @staticmethod
    def test_1():
        chi = 1
        sigma = 0.75
        n = 3
        np.testing.assert_almost_equal(mdmu.compute_v(chi, sigma, n), np.array([[0.460658866],
                                                                                  [0.2365101478],
                                                                                  [0.03200816784]]))

    @staticmethod
    def test_2():
        chi = 2
        sigma = 0.75
        n = 3
        np.testing.assert_almost_equal(mdmu.compute_v(chi, sigma, n), np.array([[0.2365101478],
                                                                                  [0.460658866],
                                                                                  [0.2365101478]]))
    @staticmethod
    def test_3():
        chi = 3
        sigma = 0.75
        n = 3
        np.testing.assert_almost_equal(mdmu.compute_v(chi, sigma, n), np.array([[0.03200816784],
                                                                                  [0.2365101478],
                                                                                  [0.460658866]]))


class TestComputeProbDensity:
    @staticmethod
    def test_1():
        x = np.array([[3],
                      [1]])
        R = np.array([[1, 1, 1, 1, 1],
                      [1, 1, 1, 0, 0]])
        m = np.array([[5],
                      [3]])
        s = np.array([[1.25],
                      [0]])
        np.testing.assert_almost_equal(mdmu.compute_prob_density(x, R, m, s), np.array([[0.1958563732],
                                                                                        [1/3]], dtype="float64"))

    @staticmethod
    def test_2():
        x = np.array([[3],
                      [1]])
        R = np.array([[1, 1, 125, 1, 1],
                      [41, 41, 45, 0, 0]])
        m = np.array([[129],
                      [127]])
        s = np.array([[1.25],
                      [0]])
        np.testing.assert_almost_equal(mdmu.compute_prob_density(x, R, m, s), np.array([[0.350585736],
                                                                                          [0.3228346457]], dtype="float64"))

    @staticmethod
    def test_3():
        x = np.array([[5],
                      [2]])
        R = np.array([[12, 2, 23, 34, 57],
                      [41, 41, 45, 0, 0]])
        m = np.array([[129],
                      [127]])
        s = np.array([[1.25],
                      [0.75]])
        np.testing.assert_almost_equal(mdmu.compute_prob_density(x, R, m, s), np.array([[0.2337592267],
                                                                                        [0.3088731198]], dtype="float64"))

    @staticmethod
    def test_4():
        x = np.array([[1],
                      [3]])
        R = np.array([[12, 2, 23, 34, 57],
                      [41, 41, 45, 0, 0]])
        m = np.array([[129],
                      [127]])
        s = np.array([[1.25],
                      [0]])
        np.testing.assert_almost_equal(mdmu.compute_prob_density(x, R, m, s), np.array([[0.05257765456],
                                                                                          [0.3543307087]], dtype="float64"))

class TestComputeD:
    @staticmethod
    def test_1():
        chi = 3
        sigma = 2
        n = 5
        displacement = 0.5
        direction = "-"
        d = np.array([[0.7111556336535151315989],
                     [0.2763263901682369329851],
                     [-0.2763263901682369329851],
                     [-0.7111556336535151315989],
                     [-0.9229001282564582301365]])
        np.testing.assert_almost_equal(mdmu.compute_d(chi, sigma, n, displacement, direction), d)

    @staticmethod
    def test_2():
        chi = 3
        sigma = 2
        n = 5
        displacement = 0.5
        direction = "+"
        d = np.array([[0.9229001282564582301365],
                     [0.7111556336535151315989],
                     [0.2763263901682369329851],
                     [-0.2763263901682369329851],
                     [-0.7111556336535151315989]])
        np.testing.assert_almost_equal(mdmu.compute_d(chi, sigma, n, displacement, direction), d)

    @staticmethod
    def test_3():
        chi = 3
        sigma = 0.002
        n = 5
        displacement = 0.5
        direction = "-"
        d = np.array([[1],
                     [1],
                     [-1],
                     [-1],
                     [-1]])
        np.testing.assert_almost_equal(mdmu.compute_d(chi, sigma, n, displacement, direction), d)

    @staticmethod
    def test_4():
        chi = 3
        sigma = 0.002
        n = 5
        displacement = 0.5
        direction = "+"
        d = np.array([[1],
                     [1],
                     [1],
                     [-1],
                     [-1]])
        np.testing.assert_almost_equal(mdmu.compute_d(chi, sigma, n, displacement, direction), d)

class TestComputeProp:
    @staticmethod
    def test_1():
        x = np.array([[1],
                      [2],
                      [3],
                      [4],
                      [5]])
        R = np.array([[85., 20., 19., 15., 18.],
                      [45., 59., 41., 11.,  1.],
                      [85., 13., 11.,  5., 43.],
                      [43., 12., 25., 30., 47.],
                      [71., 16., 11., 17., 42.]])
        m = np.array([[157],
                      [157],
                      [157],
                      [157],
                      [157]])
        s = np.array([[5],
                      [1],
                      [0.5],
                      [0.25],
                      [0.002]])
        neg_displacement = np.array([[0.5],
                                     [0.5],
                                     [0.5],
                                     [0.5],
                                     [0.5]])
        pos_displacement = np.array([[0.5],
                                     [0.5],
                                     [0.5],
                                     [0.5],
                                     [0.5]])

        p = np.array([[0.14185177],
                      [0.28059828],
                      [0.07566351],
                      [0.20269349],
                      [0.26751592]])

        np.testing.assert_almost_equal(mdmu.compute_prob(x, R, m, s, neg_displacement, pos_displacement), p)

    @staticmethod
    def test_2():
        x = np.array([[1, 2, 3, 4, 5],
                      [1, 2, 3, 4, 5],
                      [1, 2, 3, 4, 5],
                      [1, 2, 3, 4, 5],
                      [1, 2, 3, 4, 5],
                      [1, 2, 3, 4, 5],
                      [1, 2, 3, 4, 5],
                      [1, 2, 3, 4, 5],
                      [1, 2, 3, 4, 5],
                      [1, 2, 3, 4, 5]])
        R = np.array([[85., 13., 11.,  5., 43.],
                      [85., 13., 11.,  5., 43.],
                      [85., 13., 11.,  5., 43.],
                      [85., 13., 11.,  5., 43.],
                      [85., 13., 11.,  5., 43.],
                      [85., 13., 11.,  5., 43.],
                      [85., 13., 11.,  5., 43.],
                      [85., 13., 11.,  5., 43.],
                      [85., 13., 11.,  5., 43.],
                      [85., 13., 11.,  5., 43.]])
        m = np.array([[157],
                      [157],
                      [157],
                      [157],
                      [157],
                      [157],
                      [157],
                      [157],
                      [157],
                      [157]])
        s = np.array([[100],
                      [10],
                      [5],
                      [1],
                      [0.5],
                      [0.25],
                      [0.1],
                      [0.01],
                      [0.000001],
                      [0]])
        neg_displacement = np.array([[0.5],
                                     [0.5],
                                     [0.5],
                                     [0.5],
                                     [0.5],
                                     [0.5],
                                     [0.5],
                                     [0.5],
                                     [0.5],
                                     [0.5]])
        pos_displacement = np.array([[0.5],
                                     [0.5],
                                     [0.5],
                                     [0.5],
                                     [0.5],
                                     [0.5],
                                     [0.5],
                                     [0.5],
                                     [0.5],
                                     [0.5]])

        p_expected = np.array([[0.03891099769686891, 0.03925077211214341, 0.03921158439516592, 0.03879469956081954, 0.03801225387390504],
                               [0.10326065328039179, 0.10881013627005537, 0.10650868107860202, 0.09718556008630744, 0.08285896525952723],
                               [0.1298078293126285, 0.136507557722485, 0.1281206203579689, 0.11020769265202696, 0.08783508182185196],
                               [0.23183075051329086, 0.1830836727855667, 0.10394780142019805, 0.10359158524623, 0.11744065218429754],
                               [0.30142763575406983, 0.1799235958782577, 0.07566351446093961, 0.09470558523442066, 0.15084307838984154],
                               [0.3827288128460967, 0.15275820348425706, 0.06696707691944129, 0.0759586368948514, 0.19208295791807037],
                               [0.4844781114970789, 0.1081816867190897, 0.06861415912715985, 0.04779984380069729, 0.24451739640151737],
                               [0.5414009872337785, 0.08280267557713396, 0.07006368696429116, 0.031847214093434155, 0.27388520242816994],
                               [0.5414012738853503, 0.08280254777070065, 0.07006369426751592, 0.03184713375796178, 0.27388535031847133],
                               [0.5414012739, 0.08280254777, 0.07006369427, 0.03184713376, 0.2738853503]])

        p_actual = np.zeros(p_expected.shape)
        for i in range(x.shape[1]):
            p_actual[:, i] = mdmu.compute_prob(x[:, i].reshape(-1, 1), R, m, s, neg_displacement, pos_displacement).reshape(-1,)

        np.testing.assert_almost_equal(p_actual, p_expected)
        assert all(np.sum(p_actual, axis=1) <= 1)


class TestComputeMaxDistance:
    # ----------------------------- Euclidean ------------------------------- #
    @staticmethod
    def test_1():
        q = np.array([[1]])
        distance_measure = 1
        assert odu.compute_max_distance(q, distance_measure) == 0

    @staticmethod
    def test_2():
        q = np.array([[5]])
        distance_measure = 1
        assert odu.compute_max_distance(q, distance_measure) == 4

    @staticmethod
    def test_3():
        q = np.array([[1], [5]])
        distance_measure = 1
        assert odu.compute_max_distance(q, distance_measure) == 4

    @staticmethod
    def test_4():
        q = np.array([[4], [5]])
        distance_measure = 1
        assert odu.compute_max_distance(q, distance_measure) == 5

    # ----------------------------- City-block ------------------------------ #
    @staticmethod
    def test_5():
        q = np.array([[1]])
        distance_measure = 2
        assert odu.compute_max_distance(q, distance_measure) == 0

    @staticmethod
    def test_6():
        q = np.array([[5]])
        distance_measure = 2
        assert odu.compute_max_distance(q, distance_measure) == 4

    @staticmethod
    def test_7():
        q = np.array([[1], [5]])
        distance_measure = 2
        assert odu.compute_max_distance(q, distance_measure) == 4

    @staticmethod
    def test_8():
        q = np.array([[4], [5]])
        distance_measure = 2
        assert odu.compute_max_distance(q, distance_measure) == 7

    # ------------------------------- Cosine -------------------------------- #
    @staticmethod
    def test_9():
        q = np.array([[1]])
        distance_measure = 3
        assert odu.compute_max_distance(q, distance_measure) == compute_cosine_max_distance_brute_force(q)

    @staticmethod
    def test_10():
        q = np.array([[5]])
        distance_measure = 3
        assert odu.compute_max_distance(q, distance_measure) == compute_cosine_max_distance_brute_force(q)

    @staticmethod
    def test_11():
        q = np.array([[1], [5]])
        distance_measure = 3
        assert odu.compute_max_distance(q, distance_measure) == pytest.approx(compute_cosine_max_distance_brute_force(q))

    @staticmethod
    def test_12():
        q = np.array([[4], [5]])
        distance_measure = 3
        assert odu.compute_max_distance(q, distance_measure) == pytest.approx(compute_cosine_max_distance_brute_force(q))

    @staticmethod
    def test_13():
        q = np.array([[5], [3], [3]])
        distance_measure = 3
        assert odu.compute_max_distance(q, distance_measure) == pytest.approx(compute_cosine_max_distance_brute_force(q))

    @staticmethod
    def test_14():
        q = np.array([[5], [4], [3]])
        distance_measure = 3
        assert odu.compute_max_distance(q, distance_measure) == pytest.approx(compute_cosine_max_distance_brute_force(q))


class TestComputeDistances:
    # ----------------------------- Euclidean ------------------------------- #
    @staticmethod
    def test_1():
        x_new = np.array([[1]])
        X_train = np.array([[1]])
        distance_measure = 1
        assert odu.compute_distances(x_new, X_train, distance_measure, False) == np.array([0])

    @staticmethod
    def test_2():
        x_new = np.array([[1]])
        X_train = np.array([[3]])
        distance_measure = 1
        assert odu.compute_distances(x_new, X_train, distance_measure, False) == np.array([2])

    @staticmethod
    def test_3():
        x_new = np.array([[1],
                          [3]])
        X_train = np.array([[1],
                            [3]])
        distance_measure = 1
        assert odu.compute_distances(x_new, X_train, distance_measure, False) == np.array([0])

    @staticmethod
    def test_4():
        x_new = np.array([[3],
                          [1]])
        X_train = np.array([[1],
                            [3]])
        distance_measure = 1
        assert odu.compute_distances(x_new, X_train, distance_measure, False) == np.array([2 * (2 ** (1/2.0))])

    @staticmethod
    def test_5():
        x_new = np.array([[3],
                          [1]])
        X_train = np.array([[3, 3],
                            [1, 1]])
        distance_measure = 1
        assert np.all(odu.compute_distances(x_new, X_train, distance_measure, False) == np.array([[0, 0]]))

    @staticmethod
    def test_6():
        x_new = np.array([[3],
                          [1]])
        X_train = np.array([[3, 1],
                            [1, 3]])
        distance_measure = 1
        assert np.all(odu.compute_distances(x_new, X_train, distance_measure, False) == np.array([[0, 2*(2**(1/2.0))]]))

    @staticmethod
    def test_7():
        x_new = np.array([[3],
                          [1],
                          [4],
                          [2]])
        X_train = np.array([[3, 2, 1, 3, 4, 5, 2, 1, 3, 2],
                            [1, 1, 1, 3, 2, 1, 3, 4, 5, 2],
                            [3, 2, 4, 2, 1, 4, 2, 3, 4, 4],
                            [2, 3, 2, 3, 4, 5, 1, 2, 3, 2]])
        distance_measure = 1

        delta = x_new - X_train
        distances = np.linalg.norm(delta, 2, axis=0)

        assert np.all(odu.compute_distances(x_new, X_train, distance_measure, False) == distances.reshape(1, -1))

    # ----------------------------- City-block ------------------------------ #
    @staticmethod
    def test_8():
        x_new = np.array([[1]])
        X_train = np.array([[1]])
        distance_measure = 2
        assert odu.compute_distances(x_new, X_train, distance_measure, False) == np.array([0])

    @staticmethod
    def test_9():
        x_new = np.array([[1]])
        X_train = np.array([[3]])
        distance_measure = 2
        assert odu.compute_distances(x_new, X_train, distance_measure, False) == np.array([2])

    @staticmethod
    def test_10():
        x_new = np.array([[1],
                          [3]])
        X_train = np.array([[1],
                            [3]])
        distance_measure = 2
        assert odu.compute_distances(x_new, X_train, distance_measure, False) == np.array([0])

    @staticmethod
    def test_11():
        x_new = np.array([[3],
                          [1]])
        X_train = np.array([[1],
                            [3]])
        distance_measure = 2
        assert odu.compute_distances(x_new, X_train, distance_measure, False) == np.array([4])

    @staticmethod
    def test_12():
        x_new = np.array([[3],
                          [1]])
        X_train = np.array([[3, 3],
                            [1, 1]])
        distance_measure = 2
        assert np.all(odu.compute_distances(x_new, X_train, distance_measure, False) == np.array([[0, 0]]))

    @staticmethod
    def test_13():
        x_new = np.array([[3],
                          [1]])
        X_train = np.array([[3, 1],
                            [1, 3]])
        distance_measure = 2
        assert np.all(odu.compute_distances(x_new, X_train, distance_measure, False) == np.array([[0, 4]]))

    @staticmethod
    def test_14():
        x_new = np.array([[3],
                          [1],
                          [4],
                          [2]])
        X_train = np.array([[3, 2, 1, 3, 4, 5, 2, 1, 3, 2],
                            [1, 1, 1, 3, 2, 1, 3, 4, 5, 2],
                            [3, 2, 4, 2, 1, 4, 2, 3, 4, 4],
                            [2, 3, 2, 3, 4, 5, 1, 2, 3, 2]])
        distance_measure = 2

        delta = x_new - X_train
        distances = np.sum(np.abs(delta), axis=0)

        assert np.all(odu.compute_distances(x_new, X_train, distance_measure, False) == distances.reshape(1, -1))

    # ------------------------------- Cosine -------------------------------- #
    @staticmethod
    def test_15():
        x_new = np.array([[1]])
        X_train = np.array([[1]])
        distance_measure = 3
        np.testing.assert_almost_equal(odu.compute_distances(x_new, X_train, distance_measure, False), np.array([[0]]))

    @staticmethod
    def test_16():
        x_new = np.array([[1]])
        X_train = np.array([[3]])
        distance_measure = 3
        np.testing.assert_almost_equal(odu.compute_distances(x_new, X_train, distance_measure, False), np.array([[0]]))

    @staticmethod
    def test_17():
        x_new = np.array([[1],
                          [3]])
        X_train = np.array([[1],
                            [3]])
        distance_measure = 3
        np.testing.assert_almost_equal(odu.compute_distances(x_new, X_train, distance_measure, False), np.array([[0]]))

    @staticmethod
    def test_18():
        x_new = np.array([[3],
                          [1]])
        X_train = np.array([[1],
                            [3]])
        distance_measure = 3

        np.testing.assert_almost_equal(odu.compute_distances(x_new, X_train, distance_measure, False), np.array([[0.4]]))

    @staticmethod
    def test_19():
        x_new = np.array([[3],
                          [1]])
        X_train = np.array([[3, 3],
                            [1, 1]])
        distance_measure = 3
        np.testing.assert_almost_equal(odu.compute_distances(x_new, X_train, distance_measure, False), np.array([[0, 0]]))

    @staticmethod
    def test_20():
        x_new = np.array([[3],
                          [1]])
        X_train = np.array([[3, 1],
                            [1, 3]])
        distance_measure = 3
        np.testing.assert_almost_equal(odu.compute_distances(x_new, X_train, distance_measure, False), np.array([[0, 0.4]]))

    @staticmethod
    def test_21():
        x_new = np.array([[3],
                          [1]])
        X_train = np.array([[2, 1],
                            [1, 1]])
        distance_measure = 3
        np.testing.assert_almost_equal(odu.compute_distances(x_new, X_train, distance_measure, False), np.array([[1 - (7/((10 ** (1/2.0)) * (5 ** (1/2.0)))), 1 - (4/(20 ** (1/2.0)))]]))

    @staticmethod
    def test_22():
        x_new = np.array([[5],
                          [5]])
        X_train = np.array([[5, 4, 3, 2, 1],
                            [5, 4, 3, 2, 1]])
        distance_measure = 3
        np.testing.assert_almost_equal(odu.compute_distances(x_new, X_train, distance_measure, False), np.array([[0, 0, 0, 0, 0]]))

    @staticmethod
    def test_23():
        x_new = np.array([[3],
                          [1],
                          [4],
                          [2]])
        X_train = np.array([[3, 2, 1, 3, 4, 5, 2, 1, 3, 2],
                            [1, 1, 1, 3, 2, 1, 3, 4, 5, 2],
                            [3, 2, 4, 2, 1, 4, 2, 3, 4, 4],
                            [2, 3, 2, 3, 4, 5, 1, 2, 3, 2]])
        distance_measure = 3

        distances = 1 - (np.dot(x_new.T, X_train)/(np.linalg.norm(x_new, 2, axis=0) * np.linalg.norm(X_train, 2, axis=0)))
        np.testing.assert_almost_equal(odu.compute_distances(x_new, X_train, distance_measure, False), distances.reshape(1, -1))


# TODO:
# class TestCumulateDistances:
#     pass
