import pytest
import numpy as np

# Import from parent directory
from ..response_measures import *
from ..utils.utils import compute_cosine_max_distance_brute_force


class TestMultimodalDistroModel:
    class TestHandleInput:
        @staticmethod
        def test_1():
            q = np.array([5, 3])
            R = np.array([[1, 1, 1, 2, 6],
                         [1, 1, 7, 0, 0]])
            m = np.array([11, 9])
            s = np.array([1.25, 0])
            x = np.array([1, 3])
            _q, _R, _m, _s, _x = mdmu.handle_input(q, R, m, s, x)
            assert np.all(_q == np.array([[5],
                                         [3]], dtype="int64"))
            assert np.all(_R == np.array([[1, 1, 1, 2, 6],
                                         [1, 1, 7, 0, 0]], dtype="int64"))
            assert np.all(_m == np.array([[11],
                                         [9]], dtype="int64"))
            assert np.all(_s == np.array([[1.25],
                                         [0]], dtype="float64"))
            assert np.all(_x == np.array([[1],
                                         [3]], dtype="int64"))

        @staticmethod
        def test_2():
            q = np.array([[5, 3],
                          [3, 5]])
            R = np.array([1, 1, 1, 2, 6])
            m = np.array([[[11, 9]],
                          [[8, 8]]])
            s = np.array([[1.25, 0]])
            x = [[1, 3]]
            with pytest.raises(ValueError):
                mdmu.handle_input(q=q)
            with pytest.raises(ValueError):
                mdmu.handle_input(m=m)
            _, _R, _, _s, _x = mdmu.handle_input(R=R, s=s, x=x)
            assert np.all(_R == np.array([[1, 1, 1, 2, 6]], dtype="int64"))
            assert np.all(_s == np.array([[1.25],
                                          [0]], dtype="float64"))
            assert np.all(_x == np.array([[1],
                                         [3]], dtype="int64"))


    class Test__init__:
        @staticmethod
        def test_1():
            q = np.array([[5],
                          [3]])
            mdm = MultimodalDistroModel(q)
            assert np.all(mdm.R == np.array([[1, 1, 1, 1, 1],
                                             [1, 1, 1, 0, 0]], dtype="int64"))
            assert np.all(mdm.m == np.array([[5],
                                             [3]], dtype="int64"))
            assert np.all(mdm.s == np.array([[1.25],
                                             [0.75]], dtype="float64"))

        @staticmethod
        def test_2():
            q = np.array([[5],
                          [3]])
            s = np.array([[None],
                          [0]])
            mdm = MultimodalDistroModel(q, s=s)
            assert np.all(mdm.R == np.array([[1, 1, 1, 1, 1],
                                             [1, 1, 1, 0, 0]], dtype="int64"))
            assert np.all(mdm.m == np.array([[5],
                                             [3]], dtype="int64"))
            assert np.all(mdm.s == np.array([[1.25],
                                             [0]], dtype="float64"))

        @staticmethod
        def test_3():
            q = np.array([[5],
                          [3]])
            s = np.array([[81],
                          [123]])
            mdm = MultimodalDistroModel(q, s=s)
            assert np.all(mdm.R == np.array([[1, 1, 1, 1, 1],
                                             [1, 1, 1, 0, 0]], dtype="int64"))
            assert np.all(mdm.m == np.array([[5],
                                             [3]], dtype="int64"))
            assert np.all(mdm.s == np.array([[81],
                                             [123]], dtype="float64"))

        @staticmethod
        def test_4():
            q = np.array([[5],
                          [3]])
            R = np.array([[1, 1, 1],
                          [1, 1, 1]])
            m = np.array([[3],
                          [3]])
            s = np.array([[None],
                          [0]])

            mdm = MultimodalDistroModel(q, R=R, m=m, s=s)
            assert np.all(mdm.R == np.array([[1, 1, 1, 1, 1],
                                             [1, 1, 1, 0, 0]], dtype="int64"))
            assert np.all(mdm.m == np.array([[5],
                                             [3]], dtype="int64"))
            assert np.all(mdm.s == np.array([[1.25],
                                             [0]], dtype="float64"))

        @staticmethod
        def test_5():
            R = np.array([[1, 1, 1, 1, 1],
                          [1, 1, 1, 0, 0]])

            mdm = MultimodalDistroModel(R=R)
            assert np.all(mdm.R == np.array([[1, 1, 1, 1, 1],
                                             [1, 1, 1, 0, 0]], dtype="int64"))
            assert np.all(mdm.m == np.array([[5],
                                             [3]], dtype="int64"))
            assert np.all(mdm.s == np.array([[1.25],
                                             [0.75]], dtype="float64"))

        @staticmethod
        def test_6():
            R = np.array([[1, 1, 1, 1, 1],
                          [1, 1, 1, 0, 0]])
            m = np.array([[5],
                          [3]])
            s = np.array([[None],
                          [0]])

            mdm = MultimodalDistroModel(R=R, m=m, s=s)
            assert np.all(mdm.R == np.array([[1, 1, 1, 1, 1],
                                             [1, 1, 1, 0, 0]], dtype="int64"))
            assert np.all(mdm.m == np.array([[5],
                                             [3]], dtype="int64"))
            assert np.all(mdm.s == np.array([[1.25],
                                             [0]], dtype="float64"))


    class Test_CreateDistro:
        @staticmethod
        def test_1():
            q = np.array([[1]], dtype="int64")
            s = None
            R, m, s = MultimodalDistroModel._create_distro(q, s)
            assert np.all(R ==  np.array([[1]], dtype="int64"))
            assert np.all(m == np.array([[1]], dtype="int64"))
            assert np.all(s == np.array([[0.25]], dtype="float64"))

        @staticmethod
        def test_2():
            q = np.array([[5],
                          [5]], dtype="int64")
            s = None
            R, m, s = MultimodalDistroModel._create_distro(q, s)
            assert np.all(R ==  np.array([[1, 1, 1, 1, 1],
                                          [1, 1, 1, 1, 1]], dtype="int64"))
            assert np.all(m == np.array([[5],
                                         [5]], dtype="int64"))
            assert np.all(s == np.array([[1.25],
                                         [1.25]], dtype="float64"))

        @staticmethod
        def test_3():
            q = np.array([[5],
                          [5],
                          [3],
                          [2],
                          [7]], dtype="int64")
            s = None
            R, m, s = MultimodalDistroModel._create_distro(q, s)
            assert np.all(R ==  np.array([[1, 1, 1, 1, 1, 0, 0],
                                          [1, 1, 1, 1, 1, 0, 0],
                                          [1, 1, 1, 0, 0, 0, 0],
                                          [1, 1, 0, 0, 0, 0, 0],
                                          [1, 1, 1, 1, 1, 1, 1]], dtype="int64"))
            assert np.all(m == np.array([[5],
                                         [5],
                                         [3],
                                         [2],
                                         [7]], dtype="int64"))
            assert np.all(s == np.array([[1.25],
                                         [1.25],
                                         [0.75],
                                         [0.5],
                                         [1.75]], dtype="float64"))
        @staticmethod
        def test_4():
            q = np.array([[5],
                          [5],
                          [3],
                          [2],
                          [7]], dtype="int64")
            s = np.array([[1.25],
                          [1.25],
                          [0.75],
                          [0],
                          [None]], dtype="float64")
            R, m, s = MultimodalDistroModel._create_distro(q, s)
            assert np.all(R ==  np.array([[1, 1, 1, 1, 1, 0, 0],
                                          [1, 1, 1, 1, 1, 0, 0],
                                          [1, 1, 1, 0, 0, 0, 0],
                                          [1, 1, 0, 0, 0, 0, 0],
                                          [1, 1, 1, 1, 1, 1, 1]], dtype="int64"))
            assert np.all(m == np.array([[5],
                                         [5],
                                         [3],
                                         [2],
                                         [7]], dtype="int64"))
            assert np.all(s == np.array([[1.25],
                                         [1.25],
                                         [0.75],
                                         [0],
                                         [1.75]], dtype="float64"))

        @staticmethod
        def test_4():
            q = np.array([[5],
                          [5],
                          [3],
                          [2],
                          [7]], dtype="int64")
            s = np.array([[80],
                          [None],
                          [0.75],
                          [0],
                          [None]], dtype="float64")
            R, m, s = MultimodalDistroModel._create_distro(q, s)
            assert np.all(R ==  np.array([[1, 1, 1, 1, 1, 0, 0],
                                          [1, 1, 1, 1, 1, 0, 0],
                                          [1, 1, 1, 0, 0, 0, 0],
                                          [1, 1, 0, 0, 0, 0, 0],
                                          [1, 1, 1, 1, 1, 1, 1]], dtype="int64"))
            assert np.all(m == np.array([[5],
                                         [5],
                                         [3],
                                         [2],
                                         [7]], dtype="int64"))
            assert np.all(s == np.array([[80],
                                         [1.25],
                                         [0.75],
                                         [0],
                                         [1.75]], dtype="float64"))
    class TestUpdateDistro:
        @staticmethod
        def test_1():
            q = np.array([[5],
                          [3]])
            mdm = MultimodalDistroModel(q)
            x = np.array([[1],
                          [3]], dtype="int64")
            mdm.update_distro(x)
            assert np.all(mdm.R == np.array([[2, 1, 1, 1, 1],
                                             [1, 1, 2, 0, 0]], dtype="int64"))
            assert np.all(mdm.m == np.array([[6],
                                             [4]], dtype="int64"))
            assert np.all(mdm.s == np.array([[1.25],
                                             [0.75]], dtype="float64"))

        @staticmethod
        def test_2():
            q = np.array([[5],
                          [3]])
            s = np.array([[None],
                          [0]])
            mdm = MultimodalDistroModel(q, s=s)
            x = np.array([[3],
                          [2]], dtype="int64")
            mdm.update_distro(x)
            assert np.all(mdm.R == np.array([[1, 1, 2, 1, 1],
                                             [1, 2, 1, 0, 0]], dtype="int64"))
            assert np.all(mdm.m == np.array([[6],
                                             [4]], dtype="int64"))
            assert np.all(mdm.s == np.array([[1.25],
                                             [0]], dtype="float64"))


        @staticmethod
        def test_3():
            q = np.array([[5],
                          [3]])
            s = np.array([[None],
                          [0]])
            mdm = MultimodalDistroModel(q, s=s)
            x = np.array([[3],
                          [0]], dtype="int64")
            mdm.update_distro(x)
            assert np.all(mdm.R == np.array([[1, 1, 2, 1, 1],
                                             [1, 1, 1, 0, 0]], dtype="int64"))
            assert np.all(mdm.m == np.array([[6],
                                             [3]], dtype="int64"))
            assert np.all(mdm.s == np.array([[1.25],
                                             [0]], dtype="float64"))


class TestComputeScore:
    @staticmethod
    def test_1():
        R = np.array([[56, 25, 11]])
        mdm = MultimodalDistroModel(R=R)
        x = np.array([1])
        assert mdm.compute_score(x) == pytest.approx(1)

    @staticmethod
    def test_2():
        R = np.array([[56, 25, 11]])
        mdm = MultimodalDistroModel(R=R)
        x = np.array([2])
        assert mdm.compute_score(x) == pytest.approx(0.8534362614917674)

    @staticmethod
    def test_3():
        R = np.array([[56, 25, 11]])
        mdm = MultimodalDistroModel(R=R)
        x = np.array([3])
        assert mdm.compute_score(x) == pytest.approx(0.3983707249231411)

    @staticmethod
    def test_4():
        R = np.array([[56, 25, 11],
                      [47, 35, 10]])
        mdm = MultimodalDistroModel(R=R)
        x = np.array([[1],
                      [1]])
        assert mdm.compute_score(x) == pytest.approx(1)

    @staticmethod
    def test_5():
        R = np.array([[56, 25, 11],
                      [47, 35, 10]])
        mdm = MultimodalDistroModel(R=R)
        x = np.array([[1],
                      [2]])
        assert mdm.compute_score(x) == pytest.approx(0.9896524128008792)

    @staticmethod
    def test_6():
        R = np.array([[56, 25, 11],
                      [47, 35, 10]])
        mdm = MultimodalDistroModel(R=R)
        x = np.array([[1],
                      [3]])
        assert mdm.compute_score(x) == pytest.approx(0.7454676293303905)

    @staticmethod
    def test_7():
        R = np.array([[56, 25, 11],
                      [47, 35, 10]])
        mdm = MultimodalDistroModel(R=R)
        x = np.array([[2],
                      [1]])
        assert mdm.compute_score(x) == pytest.approx(0.9245861025962849)

    @staticmethod
    def test_8():
        R = np.array([[56, 25, 11],
                      [47, 35, 10]])
        mdm = MultimodalDistroModel(R=R)
        x = np.array([[2],
                      [2]])
        assert mdm.compute_score(x) == pytest.approx(0.9142385153971645)

    @staticmethod
    def test_9():
        R = np.array([[56, 25, 11],
                      [47, 35, 10]])
        mdm = MultimodalDistroModel(R=R)
        x = np.array([[2],
                      [3]])
        assert mdm.compute_score(x) == pytest.approx(0.6700537319266755)

    @staticmethod
    def test_10():
        R = np.array([[56, 25, 11],
                      [47, 35, 10]])
        mdm = MultimodalDistroModel(R=R)
        x = np.array([[3],
                      [1]])
        assert mdm.compute_score(x) == pytest.approx(0.6904336032396641)

    @staticmethod
    def test_11():
        R = np.array([[56, 25, 11],
                      [47, 35, 10]])
        mdm = MultimodalDistroModel(R=R)
        x = np.array([[3],
                      [2]])
        assert mdm.compute_score(x) == pytest.approx(0.6800860160405434)

    @staticmethod
    def test_12():
        R = np.array([[56, 25, 11],
                      [47, 35, 10]])
        mdm = MultimodalDistroModel(R=R)
        x = np.array([[3],
                      [3]])
        assert mdm.compute_score(x) == pytest.approx(0.4359012325700547)


    @staticmethod
    def test_13():
        R = np.array([[56, 25, 11],
                      [47, 35, 10]])
        mdm = MultimodalDistroModel(R=R)
        x = np.array([[3],
                      [0]])
        assert mdm.compute_score(x) == pytest.approx(0.3983707249231411)


class TestAutoAnswer:
    @staticmethod
    def test_1():
        R = np.array([[1, 3, 7],
                      [1, 9, 0]])
        mdm = MultimodalDistroModel(R=R)
        y = np.array([[1],
                      [1]])
        x = mdm.auto_answer(y)
        possible_answers = [np.array([[1], [1]]),
                            np.array([[1], [2]]),
                            np.array([[2], [1]]),
                            np.array([[2], [2]]),
                            np.array([[3], [1]]),
                            np.array([[3], [2]])]
        for answer in possible_answers:
            if np.all(x == answer):
                assert 1 == 1
                return
        assert 1 == 0

    @staticmethod
    def test_2():
        R = np.array([[1, 3, 7, 3],
                      [1, 9, 0, 0],
                      [9, 1, 3, 0]])
        mdm = MultimodalDistroModel(R=R)
        y = np.array([[1],
                      [0],
                      [1]])
        x = mdm.auto_answer(y)
        possible_answers = [np.array([[1], [0], [1]]),
                            np.array([[1], [0], [2]]),
                            np.array([[1], [0], [3]]),
                            np.array([[2], [0], [1]]),
                            np.array([[2], [0], [2]]),
                            np.array([[2], [0], [3]]),
                            np.array([[3], [0], [1]]),
                            np.array([[3], [0], [2]]),
                            np.array([[3], [0], [3]]),
                            np.array([[4], [0], [1]]),
                            np.array([[4], [0], [2]]),
                            np.array([[4], [0], [3]])]

        for answer in possible_answers:
            if np.all(x == answer):
                assert 1 == 1
                return
        print(x)
        assert 1 == 0


    @staticmethod
    def test_3():
        R = np.array([[1, 3, 7, 3],
                      [1, 9, 0, 0],
                      [9, 1, 3, 0]])
        mdm = MultimodalDistroModel(R=R)
        y = np.array([[1],
                      [0],
                      [1]])
        x, probs = mdm.auto_answer(y, return_probs=True)
        possible_answers = [np.array([[1], [0], [1]]),
                            np.array([[1], [0], [2]]),
                            np.array([[1], [0], [3]]),
                            np.array([[2], [0], [1]]),
                            np.array([[2], [0], [2]]),
                            np.array([[2], [0], [3]]),
                            np.array([[3], [0], [1]]),
                            np.array([[3], [0], [2]]),
                            np.array([[3], [0], [3]]),
                            np.array([[4], [0], [1]]),
                            np.array([[4], [0], [2]]),
                            np.array([[4], [0], [3]])]
        possible_probs = [np.array([[0.1107307], [0.], [0.32968858]]),
                          np.array([[0.1107307], [0.], [0.25530209]]),
                          np.array([[0.1107307], [0.], [0.14663731]]),
                          np.array([[0.23317215], [0.], [0.32968858]]),
                          np.array([[0.23317215], [0.], [0.25530209]]),
                          np.array([[0.23317215], [0.], [0.14663731]]),
                          np.array([[0.29938957], [0.], [0.32968858]]),
                          np.array([[0.29938957], [0.], [0.25530209]]),
                          np.array([[0.29938957], [0.], [0.14663731]]),
                          np.array([[0.21633263], [0.], [0.32968858]]),
                          np.array([[0.21633263], [0.], [0.25530209]]),
                          np.array([[0.21633263], [0.], [0.14663731]])]

        for answer, probabilities in zip(possible_answers, possible_probs):
            if np.all((x == answer) & np.isclose(probs, probabilities)):
                assert 1 == 1
                return

        print(x, probs)
        assert 1 == 0


class TestDependencyMap:
    class Test__init__:
        @staticmethod
        def test_1():
            M = np.array([[1, 2, 3],
                          [2, 3, 1],
                          [3, 1, 2]])

            assert np.all(DependencyMap(M=M).M == M)

        @staticmethod
        def test_2():
            shape = (48, 10)

            assert np.all(DependencyMap(shape=shape).M == np.zeros(shape))

        @staticmethod
        def test_3():
            M = np.array([[1, 2, 3],
                          [2, 3, 1],
                          [3, 1, 2]])
            shape = (48, 10)

            assert np.all(DependencyMap(M=M).M == M)

    class TestUpdate:
        @staticmethod
        def test_1():
            q_0 = np.array([[5],
                            [4],
                            [1],
                            [2],
                            [3]])
            q_1 = np.array([[5, 3, 1, 5, 2]])
            shape = (q_0.shape[0], q_1.shape[1])
            expected_M = np.array([[1, 0, 0, 1, 0],
                                   [0, 0, 0, 0, 0],
                                   [0, 0, 1, 0, 0],
                                   [0, 0, 0, 0, 1],
                                   [0, 1, 0, 0, 0]])

            m = DependencyMap(shape=shape)
            m.update(q_0, q_1)

            assert np.all(m.M == expected_M)

        @staticmethod
        def test_2():
            q_0 = np.array([[5],
                            [4],
                            [1],
                            [2],
                            [3]])
            q_1 = np.array([[5, 3, 1, 5, 2]])
            M = np.array([[2, 3, 4, 5, 6],
                          [3, 2, 1, 4, 5],
                          [1, 2, 4, 5, 6],
                          [5, 3, 1, 2, 4],
                          [7, 2, 3, 4, 5]])
            expected_M = np.array([[3, 3, 4, 6, 6],
                                   [3, 2, 1, 4, 5],
                                   [1, 2, 5, 5, 6],
                                   [5, 3, 1, 2, 5],
                                   [7, 3, 3, 4, 5]])

            m = DependencyMap(M=M)
            m.update(q_0, q_1)

            assert np.all(m.M == expected_M)

    class TestM_to_R:
        @staticmethod
        def test_1():
            M = np.array([[39, 27, 16, 3],
                          [19, 27, 35, 26],
                          [2, 3, 9, 18]])
            dm = DependencyMap(M=M)
            q_1 = np.array([[5, 4, 3, 3]])
            R_width = 5
            m = np.array([[50]] * 3)

            R = dm.M_to_R(q_1, R_width, m)

            assert np.all(R == np.array([[0, 0, 0, 27, 39],
                                         [0, 0, 35, 27, 0],
                                         [0, 0, 18, 0, 0]]))


class TestTimeResponseMeasure:
    @staticmethod
    def test_1():
        respondent_time = 0
        SURVEY_READING_TIME = 5
        assert time_response_measure(respondent_time, SURVEY_READING_TIME) == 0

    @staticmethod
    def test_2():
        respondent_time = 5
        SURVEY_READING_TIME = 5
        assert time_response_measure(respondent_time, SURVEY_READING_TIME) == 1

    @staticmethod
    def test_3():
        respondent_time = 4
        SURVEY_READING_TIME = 5
        assert time_response_measure(respondent_time, SURVEY_READING_TIME) == float(respondent_time/SURVEY_READING_TIME)

    @staticmethod
    def test_4():
        respondent_time = 6
        SURVEY_READING_TIME = 5
        assert time_response_measure(respondent_time, SURVEY_READING_TIME) == 1

    @staticmethod
    def test_5():
        respondent_time = -1
        SURVEY_READING_TIME = 5
        with pytest.raises(ValueError):
            time_response_measure(respondent_time, SURVEY_READING_TIME)

    @staticmethod
    def test_6():
        respondent_time = 4
        SURVEY_READING_TIME = 0
        with pytest.raises(ValueError):
            time_response_measure(respondent_time, SURVEY_READING_TIME)

    @staticmethod
    def test_7():
        respondent_time = 4
        SURVEY_READING_TIME = -1
        with pytest.raises(ValueError):
            time_response_measure(respondent_time, SURVEY_READING_TIME)


class TestODResponseMeasure:
    # Question 1: Rate Salah's pace.
    #   Options: (1) Very slow. (2) Slow. (3) Moderate. (4) Fast. (5) Very fast.
    # Question 2: Rate Salah's defensive awareness.
    #   Options: (1) Very unaware. (2) Unaware. (3) Moderate. (4) Aware. (5) Very aware.

    # ----------------------------- Value errors ---------------------------- #

    @staticmethod
    def test_1():
        """
         Raise value error for q, empty vector.
        """
        q = np.array([])
        x_new = np.array([[2]])
        X_train = np.array([[1, 2]])
        k = 1
        distance_measure = 1
        with pytest.raises(ValueError):
            od_response_measure(q, x_new, X_train, k, distance_measure)

    @staticmethod
    def test_2():
        """
         Raise value error for q, wrong dimensions.
        """
        q = np.array([[1, 2],
                      [3, 4]])
        x_new = np.array([[2]])
        X_train = np.array([[1, 2]])
        k = 1
        distance_measure = 1
        with pytest.raises(ValueError):
            od_response_measure(q, x_new, X_train, k, distance_measure)

    @staticmethod
    def test_3():
        """
         Raise value error for x_new, empty value.
        """
        q = np.array([[2]])
        x_new = np.array([])
        X_train = np.array([[1, 2]])
        k = 1
        distance_measure = 1
        with pytest.raises(ValueError):
            od_response_measure(q, x_new, X_train, k, distance_measure)

    @staticmethod
    def test_4():
        """
         Raise value error for distance_measure, wrong value.
        """
        q = np.array([[2]])
        x_new = np.array([[1]])
        X_train = np.array([[1, 2]])
        k = 1
        distance_measure = -1
        with pytest.raises(ValueError):
            od_response_measure(q, x_new, X_train, k, distance_measure)

    @staticmethod
    def test_5():
        """
         Raise value error for distance_measure, wrong value.
        """
        q = np.array([[2]])
        x_new = np.array([[1]])
        X_train = np.array([[1, 2]])
        k = 1
        distance_measure = 901
        with pytest.raises(ValueError):
            od_response_measure(q, x_new, X_train, k, distance_measure)

    @staticmethod
    def test_6():
        """
         Raise value error for (q, x_new), wrong values.
        """
        q = np.array([[2]])
        x_new = np.array([[3]])
        X_train = np.array([[1, 2]])
        k = 1
        distance_measure = 1
        with pytest.raises(ValueError):
            od_response_measure(q, x_new, X_train, k, distance_measure)

    @staticmethod
    def test_7():
        """
         Raise value error for (q, x_new), wrong dimensions.
        """
        q = np.array([[2],
                      [2]])
        x_new = np.array([[1]])
        X_train = np.array([[1, 2]])
        k = 1
        distance_measure = 1
        with pytest.raises(ValueError):
            od_response_measure(q, x_new, X_train, k, distance_measure)

    @staticmethod
    def test_8():
        """
         Raise value error for (q, x_new), wrong dimensions.
        """
        q = np.array([[2]])
        x_new = np.array([[1],
                          [1]])
        X_train = np.array([[1, 2]])
        k = 1
        distance_measure = 1
        with pytest.raises(ValueError):
            od_response_measure(q, x_new, X_train, k, distance_measure)

    @staticmethod
    def test_9():
        """
         Raise value error for (q, X_train), wrong value.
        """
        q = np.array([[2]])
        x_new = np.array([[1]])
        X_train = np.array([[1, 3]])
        k = 1
        distance_measure = 1
        with pytest.raises(ValueError):
            od_response_measure(q, x_new, X_train, k, distance_measure)

    @staticmethod
    def test_10():
        """
         Raise value error for (q, X_train), wrong dimensions.
        """
        q = np.array([[2],
                      [2]])
        x_new = np.array([[1]])
        X_train = np.array([[1, 2]])
        k = 1
        distance_measure = 1
        with pytest.raises(ValueError):
            od_response_measure(q, x_new, X_train, k, distance_measure)

    @staticmethod
    def test_11():
        """
         Raise value error for (q, X_train), wrong dimensions.
        """
        q = np.array([[2]])
        x_new = np.array([[1]])
        X_train = np.array([[1, 2],
                            [2, 1]])
        k = 1
        distance_measure = 1
        with pytest.raises(ValueError):
            od_response_measure(q, x_new, X_train, k, distance_measure)

    @staticmethod
    def test_12():
        """
         Raise value error for k, wrong value.
        """
        q = np.array([[2]])
        x_new = np.array([[1]])
        X_train = np.array([[1, 3]])
        k = -1
        distance_measure = 1
        with pytest.raises(ValueError):
            od_response_measure(q, x_new, X_train, k, distance_measure)

    @staticmethod
    def test_13():
        """
         Raise value error for q, wrong value.
        """
        q = np.array([0])
        x_new = np.array([[2]])
        X_train = np.array([[1, 2]])
        k = 1
        distance_measure = 1
        with pytest.raises(ValueError):
            od_response_measure(q, x_new, X_train, k, distance_measure)

    # ------ Variate (distance_measure -> k -> X_train -> x_new -> q) ------- #

    @staticmethod
    def test_14():
        """
        Trivial case 1: - one question with one answer.
                        - No previous answers.
        For aribtrary valid value of k and any distance measure, result must be 1,
        as you don't have information yet.
        """
        q = np.array([[1]])
        x_new = np.array([[1]])
        X_train = np.array([[]])
        k = 1
        distance_measure = 1
        assert od_response_measure(q, x_new, X_train, k, distance_measure) == 1

    @staticmethod
    def test_15():
        """
        Trivial case 1: - one question with one answer.
                        - No previous answers.
        For aribtrary valid value of k and any distance measure, result must be 1,
        as you don't have information yet.
        """
        q = np.array([[1]])
        x_new = np.array([[1]])
        X_train = np.array([[]])
        k = 124
        distance_measure = 2
        assert od_response_measure(q, x_new, X_train, k, distance_measure) == 1

    @staticmethod
    def test_16():
        """
        Trivial case 1: - one question with one answer.
                        - No previous answers.
        For aribtrary valid value of k and any distance measure, result must be 1,
        as you don't have information yet.
        """
        q = np.array([[1]])
        x_new = np.array([[1]])
        X_train = np.array([[]])
        k = 8124
        distance_measure = 3
        assert od_response_measure(q, x_new, X_train, k, distance_measure) == 1

    @staticmethod
    def test_17():
        """
        Trivial case 2: - one question with one answer.
                        - 1 previous answer.
        For aribtrary valid value of k and any distance measure, result must be 1,
        as only one answer is allowed.
        """
        q = np.array([[1]])
        x_new = np.array([[1]])
        X_train = np.array([[1]])
        k = 124
        distance_measure = 1
        assert od_response_measure(q, x_new, X_train, k, distance_measure) == 1

    @staticmethod
    def test_18():
        """
        Trivial case 2: - one question with one answer.
                        - 1 previous answer.
        For aribtrary valid value of k and any distance measure, result must be 1,
        as only one answer is allowed.
        """
        q = np.array([[1]])
        x_new = np.array([[1]])
        X_train = np.array([[1]])
        k = 124
        distance_measure = 2
        assert od_response_measure(q, x_new, X_train, k, distance_measure) == 1

    @staticmethod
    def test_19():
        """
        Trivial case 2: - one question with one answer.
                        - 1 previous answer.
        For aribtrary valid value of k and any distance measure, result must be 1,
        as only one answer is allowed.
        """
        q = np.array([[1]])
        x_new = np.array([[1]])
        X_train = np.array([[1]])
        k = 124
        distance_measure = 1
        assert od_response_measure(q, x_new, X_train, k, distance_measure) == 1

    @staticmethod
    def test_20():
        """
        Trivial case 3: - one question with one answer.
                        - aribtrary number of previous answers.
        For aribtrary valid value of k and any distance measure, result must be 1,
        as only one answer is allowed.
        """
        q = np.array([[1]])
        x_new = np.array([[1]])
        X_train = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        k = 124
        distance_measure = 1
        assert od_response_measure(q, x_new, X_train, k, distance_measure) == 1

    @staticmethod
    def test_21():
        """
        Trivial case 3: - one question with one answer.
                        - aribtrary number of previous answers.
        For aribtrary valid value of k and any distance measure, result must be 1,
        as only one answer is allowed.
        """
        q = np.array([[1]])
        x_new = np.array([[1]])
        X_train = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        k = 124
        distance_measure = 2
        assert od_response_measure(q, x_new, X_train, k, distance_measure) == 1

    @staticmethod
    def test_22():
        """
        Trivial case 3: - one question with one answer.
                        - aribtrary number of previous answers.
        For aribtrary valid value of k and any distance measure, result must be 1,
        as only one answer is allowed.
        """
        q = np.array([[1]])
        x_new = np.array([[1]])
        X_train = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        k = 124
        distance_measure = 3
        assert od_response_measure(q, x_new, X_train, k, distance_measure) == 1

    @staticmethod
    def test_23():
        """
        Case 1: Question 1: What's your gender?
                 Options: (1) Male. (2) Female.
                - One question with 2 options.
                - No previous answers.
        For aribtrary valid value of k and any distance measure, result must be 1,
        as there is no information yet.
        """
        q = np.array([[2]])
        x_new = np.array([[1]])
        X_train = np.array([])
        k = 124
        distance_measure = 1
        assert od_response_measure(q, x_new, X_train, k, distance_measure) == 1

    @staticmethod
    def test_24():
        """
        Case 1: Question 1: What's your gender?
                 Options: (1) Male. (2) Female.
                - One question with 2 options.
                - No previous answers.
        For aribtrary valid value of k and any distance measure, result must be 1,
        as there is no information yet.
        """
        q = np.array([[2]])
        x_new = np.array([[1]])
        X_train = np.array([])
        k = 124
        distance_measure = 2
        assert od_response_measure(q, x_new, X_train, k, distance_measure) == 1

    @staticmethod
    def test_25():
        """
        Case 1: Question 1: What's your gender?
                 Options: (1) Male. (2) Female.
                - One question with 2 options.
                - No previous answers.
        For aribtrary valid value of k and any distance measure, result must be 1,
        as there is no information yet.
        """
        q = np.array([[2]])
        x_new = np.array([[1]])
        X_train = np.array([])
        k = 124
        distance_measure = 3
        assert od_response_measure(q, x_new, X_train, k, distance_measure) == 1

    @staticmethod
    def test_26():
        """
        Case 1: Question 1: What's your gender?
                 Options: (1) Male. (2) Female.
                - One question with 2 options.
                - One previous answer.
        For an arbitrary valid value of k and any distance meaesure, the result
        will be either 1 if the new answer matches the previous one or 0 otherwise.
        """
        q = np.array([[2]])
        x_new = np.array([[1]])
        X_train = np.array([1])
        k = 1
        distance_measure = 1
        assert od_response_measure(q, x_new, X_train, k, distance_measure) == 1

    @staticmethod
    def test_27():
        """
        Case 1: Question 1: What's your gender?
                 Options: (1) Male. (2) Female.
                - One question with 2 options.
                - One previous answer.
        For an arbitrary valid value of k and any distance meaesure, the result
        will be either 1 if the new answer matches the previous one or 0 otherwise.
        """
        q = np.array([[2]])
        x_new = np.array([[2]])
        X_train = np.array([1])
        k = 135
        distance_measure = 1
        assert od_response_measure(q, x_new, X_train, k, distance_measure) == 0

    @staticmethod
    def test_28():
        """
        Case 1: Question 1: What's your gender?
                 Options: (1) Male. (2) Female.
                - One question with 2 options.
                - One previous answer.
        For an arbitrary valid value of k and any distance meaesure, the result
        will be either 1 if the new answer matches the previous one or 0 otherwise.
        """
        q = np.array([[2]])
        x_new = np.array([[2]])
        X_train = np.array([2])
        k = 1
        distance_measure = 2
        assert od_response_measure(q, x_new, X_train, k, distance_measure) == 1

    @staticmethod
    def test_29():
        """
        Case 1: Question 1: What's your gender?
                 Options: (1) Male. (2) Female.
                - One question with 2 options.
                - One previous answer.
        For an arbitrary valid value of k and any distance meaesure, the result
        will be either 1 if the new answer matches the previous one or 0 otherwise.
        """
        q = np.array([[2]])
        x_new = np.array([[2]])
        X_train = np.array([1])
        k = 1
        distance_measure = 2
        assert od_response_measure(q, x_new, X_train, k, distance_measure) == 0

    @staticmethod
    def test_30():
        """
        Case 1: Question 1: What's your gender?
                 Options: (1) Male. (2) Female.
                - One question with 2 options.
                - One previous answer.
        For an arbitrary valid value of k and any distance meaesure, the result
        will be either 1 if the new answer matches the previous one or 0 otherwise.
        """
        q = np.array([[2]])
        x_new = np.array([[1]])
        X_train = np.array([1])
        k = 1
        distance_measure = 3
        assert od_response_measure(q, x_new, X_train, k, distance_measure) == 1

    @staticmethod
    def test_31():
        """
        Case 1: Question 1: What's your gender?
                 Options: (1) Male. (2) Female.
                - One question with 2 options.
                - One previous answer.
        For an arbitrary valid value of k and cosine distance meaesure, the result
        will always be 1 because there exists only one dimension.
        """
        q = np.array([[2]])
        x_new = np.array([[2]])
        X_train = np.array([1])
        k = 1
        distance_measure = 3
        assert od_response_measure(q, x_new, X_train, k, distance_measure) == 1

    @staticmethod
    def test_32():
        """
        Case 1: Question 1: What's your gender?
                 Options: (1) Male. (2) Female.
                - One question with 2 options.
                - Two previous answers.
                - k = 1.
                - Euclidean Distance.
        The result should be 1 if one or both of the previous answers match
        the new answer, 0 otherwise.
        """
        q = np.array([[2]])
        x_new = np.array([[1]])
        X_train = np.array([1, 2])
        k = 1
        distance_measure = 1
        assert od_response_measure(q, x_new, X_train, k, distance_measure) == 1

    @staticmethod
    def test_33():
        """
        Case 1: Question 1: What's your gender?
                 Options: (1) Male. (2) Female.
                - One question with 2 options.
                - Two previous answers.
                - k = 1.
                - Euclidean Distance.
        The result should be 1 if one or both of the previous answers match
        the new answer, 0 otherwise.
        """
        q = np.array([[2]])
        x_new = np.array([[1]])
        X_train = np.array([2, 2])
        k = 1
        distance_measure = 1
        assert od_response_measure(q, x_new, X_train, k, distance_measure) == 0

    @staticmethod
    def test_34():
        """
        Case 1: Question 1: What's your gender?
                 Options: (1) Male. (2) Female.
                - One question with 2 options.
                - Two previous answers.
                - k = 1.
                - City-block Distance.
        The result should be 1 if one or both of the previous answers match
        the new answer, 0 otherwise.
        """
        q = np.array([[2]])
        x_new = np.array([[1]])
        X_train = np.array([1, 2])
        k = 1
        distance_measure = 2
        assert od_response_measure(q, x_new, X_train, k, distance_measure) == 1

    @staticmethod
    def test_35():
        """
        Case 1: Question 1: What's your gender?
                 Options: (1) Male. (2) Female.
                - One question with 2 options.
                - Two previous answers.
                - k = 1.
                - City-block Distance.
        The result should be 1 if one or both of the previous answers match
        the new answer, 0 otherwise.
        """
        q = np.array([[2]])
        x_new = np.array([[2]])
        X_train = np.array([1, 1])
        k = 1
        distance_measure = 2
        assert od_response_measure(q, x_new, X_train, k, distance_measure) == 0

    @staticmethod
    def test_36():
        """
        Case 1: Question 1: What's your gender?
                 Options: (1) Male. (2) Female.
                - One question with 2 options.
                - Two previous answers.
                - k = 1.
                - Cosine Distance.
        The result should be 1 no matter what as one question means one axis, and
        one axis means that the angle between any two vectors is zero, and cos(0) = 1.
        """
        q = np.array([[2]])
        x_new = np.array([[1]])
        X_train = np.array([1, 2])
        k = 1
        distance_measure = 3
        assert od_response_measure(q, x_new, X_train, k, distance_measure) == 1

    @staticmethod
    def test_37():
        """
        Case 1: Question 1: What's your gender?
                 Options: (1) Male. (2) Female.
                - One question with 2 options.
                - Two previous answers.
                - k = 1.
                - Cosine Distance.
        The result should be 1 no matter what as one question means one axis, and
        one axis means that the angle between any two vectors is zero, and cos(0) = 1.
        """
        q = np.array([[2]])
        x_new = np.array([[1]])
        X_train = np.array([2, 2])
        k = 1
        distance_measure = 3
        assert od_response_measure(q, x_new, X_train, k, distance_measure) == 1

    @staticmethod
    def test_38():
        """
        Case 1: Question 1: What's your gender?
                 Options: (1) Male. (2) Female.
                - One question with 2 options.
                - Two previous answers.
                - k = 2.
                - Euclidean Distance.
        The result should be 1 if and only if the two prev answers match the new
        one. 1/2 if one of them match it. 0 if none of them match it.
        """
        q = np.array([[2]])
        x_new = np.array([[1]])
        X_train = np.array([1, 1])
        k = 2
        distance_measure = 1
        assert od_response_measure(q, x_new, X_train, k, distance_measure) == 1

    @staticmethod
    def test_39():
        """
        Case 1: Question 1: What's your gender?
                 Options: (1) Male. (2) Female.
                - One question with 2 options.
                - Two previous answers.
                - k = 2.
                - Euclidean Distance.
        The result should be 1 if and only if the two prev answers match the new
        one. 1/2 if one of them match it. 0 if none of them match it.
        """
        q = np.array([[2]])
        x_new = np.array([[1]])
        X_train = np.array([2, 1])
        k = 2
        distance_measure = 1
        assert od_response_measure(q, x_new, X_train, k, distance_measure) == 1/2.0

    @staticmethod
    def test_40():
        """
        Case 1: Question 1: What's your gender?
                 Options: (1) Male. (2) Female.
                - One question with 2 options.
                - Two previous answers.
                - k = 2.
                - Euclidean Distance.
        The result should be 1 if and only if the two prev answers match the new
        one. 1/2 if one of them match it. 0 if none of them match it.
        """
        q = np.array([[2]])
        x_new = np.array([[1]])
        X_train = np.array([1, 2])
        k = 2
        distance_measure = 1
        assert od_response_measure(q, x_new, X_train, k, distance_measure) == 1/2.0

    @staticmethod
    def test_41():
        """
        Case 1: Question 1: What's your gender?
                 Options: (1) Male. (2) Female.
                - One question with 2 options.
                - Two previous answers.
                - k = 2.
                - Euclidean Distance.
        The result should be 1 if and only if the two prev answers match the new
        one. 1/2 if one of them match it. 0 if none of them match it.
        """
        q = np.array([[2]])
        x_new = np.array([[2]])
        X_train = np.array([1, 2])
        k = 2
        distance_measure = 1
        assert od_response_measure(q, x_new, X_train, k, distance_measure) == 1/2.0

    @staticmethod
    def test_42():
        """
        Case 1: Question 1: What's your gender?
                 Options: (1) Male. (2) Female.
                - One question with 2 options.
                - Two previous answers.
                - k = 2.
                - Euclidean Distance.
        The result should be 1 if and only if the two prev answers match the new
        one. 1/2 if one of them match it. 0 if none of them match it.
        """
        q = np.array([[2]])
        x_new = np.array([[1]])
        X_train = np.array([2, 2])
        k = 2
        distance_measure = 1
        assert od_response_measure(q, x_new, X_train, k, distance_measure) == 0

    @staticmethod
    def test_43():
        """
        Case 1: Question 1: What's your gender?
                 Options: (1) Male. (2) Female.
                - One question with 2 options.
                - Two previous answers.
                - k = 2.
                - City-block Distance.
        The result should be 1 if and only if the two prev answers match the new
        one. 1/2 if one of them match it. 0 if none of them match it.
        """
        q = np.array([[2]])
        x_new = np.array([[1]])
        X_train = np.array([1, 1])
        k = 2
        distance_measure = 2
        assert od_response_measure(q, x_new, X_train, k, distance_measure) == 1

    @staticmethod
    def test_44():
        """
        Case 1: Question 1: What's your gender?
                 Options: (1) Male. (2) Female.
                - One question with 2 options.
                - Two previous answers.
                - k = 2.
                - City-block Distance.
        The result should be 1 if and only if the two prev answers match the new
        one. 1/2 if one of them match it. 0 if none of them match it.
        """
        q = np.array([[2]])
        x_new = np.array([[1]])
        X_train = np.array([2, 1])
        k = 2
        distance_measure = 2
        assert od_response_measure(q, x_new, X_train, k, distance_measure) == 1/2.0

    @staticmethod
    def test_45():
        """
        Case 1: Question 1: What's your gender?
                 Options: (1) Male. (2) Female.
                - One question with 2 options.
                - Two previous answers.
                - k = 2.
                - City-block Distance.
        The result should be 1 if and only if the two prev answers match the new
        one. 1/2 if one of them match it. 0 if none of them match it.
        """
        q = np.array([[2]])
        x_new = np.array([[1]])
        X_train = np.array([1, 2])
        k = 2
        distance_measure = 2
        assert od_response_measure(q, x_new, X_train, k, distance_measure) == 1/2.0

    @staticmethod
    def test_46():
        """
        Case 1: Question 1: What's your gender?
                 Options: (1) Male. (2) Female.
                - One question with 2 options.
                - Two previous answers.
                - k = 2.
                - City-block Distance.
        The result should be 1 if and only if the two prev answers match the new
        one. 1/2 if one of them match it. 0 if none of them match it.
        """
        q = np.array([[2]])
        x_new = np.array([[2]])
        X_train = np.array([1, 2])
        k = 2
        distance_measure = 2
        assert od_response_measure(q, x_new, X_train, k, distance_measure) == 1/2.0

    @staticmethod
    def test_47():
        """
        Case 1: Question 1: What's your gender?
                 Options: (1) Male. (2) Female.
                - One question with 2 options.
                - Two previous answers.
                - k = 2.
                - City-block Distance.
        The result should be 1 if and only if the two prev answers match the new
        one. 1/2 if one of them match it. 0 if none of them match it.
        """
        q = np.array([[2]])
        x_new = np.array([[1]])
        X_train = np.array([2, 2])
        k = 2
        distance_measure = 2
        assert od_response_measure(q, x_new, X_train, k, distance_measure) == 0

    @staticmethod
    def test_48():
        """
        Case 1: Question 1: What's your gender?
                 Options: (1) Male. (2) Female.
                - One question with 2 options.
                - Two previous answers.
                - k = 2.
                - Cosine Distance.
        The result should be 1 no matter what as one question means one axis, and
        one axis means that the angle between any two vectors is zero, and cos(0) = 1.
        """
        q = np.array([[2]])
        x_new = np.array([[1]])
        X_train = np.array([1, 1])
        k = 2
        distance_measure = 3
        assert od_response_measure(q, x_new, X_train, k, distance_measure) == 1

    @staticmethod
    def test_49():
        """
        Case 1: Question 1: What's your gender?
                 Options: (1) Male. (2) Female.
                - One question with 2 options.
                - Two previous answers.
                - k = 2.
                - Cosine Distance.
        The result should be 1 no matter what as one question means one axis, and
        one axis means that the angle between any two vectors is zero, and cos(0) = 1.
        """
        q = np.array([[2]])
        x_new = np.array([[1]])
        X_train = np.array([2, 1])
        k = 2
        distance_measure = 3
        assert od_response_measure(q, x_new, X_train, k, distance_measure) == 1

    @staticmethod
    def test_50():
        """
        Case 1: Question 1: What's your gender?
                 Options: (1) Male. (2) Female.
                - One question with 2 options.
                - Two previous answers.
                - k = 2.
                - Cosine Distance.
        The result should be 1 no matter what as one question means one axis, and
        one axis means that the angle between any two vectors is zero, and cos(0) = 1.
        """
        q = np.array([[2]])
        x_new = np.array([[1]])
        X_train = np.array([1, 2])
        k = 2
        distance_measure = 3
        assert od_response_measure(q, x_new, X_train, k, distance_measure) == 1

    @staticmethod
    def test_51():
        """
        Case 1: Question 1: What's your gender?
                 Options: (1) Male. (2) Female.
                - One question with 2 options.
                - Two previous answers.
                - k = 2.
                - Cosine Distance.
        The result should be 1 no matter what as one question means one axis, and
        one axis means that the angle between any two vectors is zero, and cos(0) = 1.
        """
        q = np.array([[2]])
        x_new = np.array([[2]])
        X_train = np.array([1, 2])
        k = 2
        distance_measure = 3
        assert od_response_measure(q, x_new, X_train, k, distance_measure) == 1

    @staticmethod
    def test_52():
        """
        Case 1: Question 1: What's your gender?
                 Options: (1) Male. (2) Female.
                - One question with 2 options.
                - Two previous answers.
                - k = 2.
                - Cosine Distance.
        The result should be 1 no matter what as one question means one axis, and
        one axis means that the angle between any two vectors is zero, and cos(0) = 1.
        """
        q = np.array([[2]])
        x_new = np.array([[1]])
        X_train = np.array([2, 2])
        k = 2
        distance_measure = 3
        assert od_response_measure(q, x_new, X_train, k, distance_measure) == 1

    # --------------------------------------------------------------------------

    @staticmethod
    def test_53():
        """
        Case 1: Question 1: What's your gender?
                 Options: (1) Male. (2) Female.
                - One question with 2 options.
                - Arbitrary number of previous answers.
                - Arbitrary value of k.
                - Euclidean Distance.
        The result should be 1 if and only if there exists k prev answers match the new
        one. 1/k if one of them match it. l/k if l of them match it. 0 if none of them match it.
        """
        q = np.array([[2]])
        x_new = np.array([[2]])
        X_train = np.array([2, 1, 2, 1, 1, 2, 2, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2])
        k = 7
        distance_measure = 1
        assert od_response_measure(q, x_new, X_train, k, distance_measure) == 1

    @staticmethod
    def test_54():
        """
        Case 1: Question 1: What's your gender?
                 Options: (1) Male. (2) Female.
                - One question with 2 options.
                - Arbitrary number of previous answers.
                - Arbitrary value of k.
                - Euclidean Distance.
        The result should be 1 if and only if there exists k prev answers match the new
        one. 1/k if one of them match it. l/k if l of them match it. 0 if none of them match it.
        """
        q = np.array([[2]])
        x_new = np.array([[2]])
        X_train = np.array([2, 1, 1, 1, 1, 2, 2, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2])
        k = 7
        distance_measure = 1
        assert od_response_measure(q, x_new, X_train, k, distance_measure) == pytest.approx(6/7.0)

    @staticmethod
    def test_55():
        """
        Case 1: Question 1: What's your gender?
                 Options: (1) Male. (2) Female.
                - One question with 2 options.
                - Arbitrary number of previous answers.
                - Arbitrary value of k.
                - Euclidean Distance.
        The result should be 1 if and only if there exists k prev answers match the new
        one. 1/k if one of them match it. l/k if l of them match it. 0 if none of them match it.
        """
        q = np.array([[2]])
        x_new = np.array([[2]])
        X_train = np.array([2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        k = 7
        distance_measure = 1
        assert od_response_measure(q, x_new, X_train, k, distance_measure) == pytest.approx(1/7.0)

    @staticmethod
    def test_56():
        """
        Case 1: Question 1: What's your gender?
                 Options: (1) Male. (2) Female.
                - One question with 2 options.
                - Arbitrary number of previous answers.
                - Arbitrary value of k.
                - Euclidean Distance.
        The result should be 1 if and only if there exists k prev answers match the new
        one. 1/k if one of them match it. l/k if l of them match it. 0 if none of them match it.
        """
        q = np.array([[2]])
        x_new = np.array([[2]])
        X_train = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        k = 7
        distance_measure = 1
        assert od_response_measure(q, x_new, X_train, k, distance_measure) == 0

    @staticmethod
    def test_57():
        """
        Case 1: Question 1: What's your gender?
                 Options: (1) Male. (2) Female.
                - One question with 2 options.
                - Arbitrary number of previous answers.
                - Arbitrary value of k.
                - City-block Distance.
        The result should be 1 if and only if there exists k prev answers match the new
        one. 1/k if one of them match it. l/k if l of them match it. 0 if none of them match it.
        """
        q = np.array([[2]])
        x_new = np.array([[2]])
        X_train = np.array([2, 1, 2, 1, 1, 2, 2, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2])
        k = 7
        distance_measure = 2
        assert od_response_measure(q, x_new, X_train, k, distance_measure) == 1

    @staticmethod
    def test_58():
        """
        Case 1: Question 1: What's your gender?
                 Options: (1) Male. (2) Female.
                - One question with 2 options.
                - Arbitrary number of previous answers.
                - Arbitrary value of k.
                - City-block Distance.
        The result should be 1 if and only if there exists k prev answers match the new
        one. 1/k if one of them match it. l/k if l of them match it. 0 if none of them match it.
        """
        q = np.array([[2]])
        x_new = np.array([[2]])
        X_train = np.array([2, 1, 1, 1, 1, 2, 2, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2])
        k = 7
        distance_measure = 2
        assert od_response_measure(q, x_new, X_train, k, distance_measure) == pytest.approx(6/7.0)

    @staticmethod
    def test_59():
        """
        Case 1: Question 1: What's your gender?
                 Options: (1) Male. (2) Female.
                - One question with 2 options.
                - Arbitrary number of previous answers.
                - Arbitrary value of k.
                - City-block Distance.
        The result should be 1 if and only if there exists k prev answers match the new
        one. 1/k if one of them match it. l/k if l of them match it. 0 if none of them match it.
        """
        q = np.array([[2]])
        x_new = np.array([[2]])
        X_train = np.array([2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        k = 7
        distance_measure = 2
        assert od_response_measure(q, x_new, X_train, k, distance_measure) == pytest.approx(1/7.0)

    @staticmethod
    def test_60():
        """
        Case 1: Question 1: What's your gender?
                 Options: (1) Male. (2) Female.
                - One question with 2 options.
                - Arbitrary number of previous answers.
                - Arbitrary value of k.
                - City-block Distance.
        The result should be 1 if and only if there exists k prev answers match the new
        one. 1/k if one of them match it. l/k if l of them match it. 0 if none of them match it.
        """
        q = np.array([[2]])
        x_new = np.array([[2]])
        X_train = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        k = 7
        distance_measure = 2
        assert od_response_measure(q, x_new, X_train, k, distance_measure) == 0

    @staticmethod
    def test_61():
        """
        Case 1: Question 1: What's your gender?
                 Options: (1) Male. (2) Female.
                - One question with 2 options.
                - Arbitrary number of previous answers.
                - Arbitrary value of k.
                - Cosine Distance.
        The result should be 1 no matter what as one question means one axis, and
        one axis means that the angle between any two vectors is zero, and cos(0) = 1.
        """
        q = np.array([[2]])
        x_new = np.array([[2]])
        X_train = np.array([2, 1, 2, 1, 1, 2, 2, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2])
        k = 7
        distance_measure = 3
        assert od_response_measure(q, x_new, X_train, k, distance_measure) == 1

    @staticmethod
    def test_62():
        """
        Case 1: Question 1: What's your gender?
                 Options: (1) Male. (2) Female.
                - One question with 2 options.
                - Arbitrary number of previous answers.
                - Arbitrary value of k.
                - Cosine Distance.
        The result should be 1 no matter what as one question means one axis, and
        one axis means that the angle between any two vectors is zero, and cos(0) = 1.
        """
        q = np.array([[2]])
        x_new = np.array([[2]])
        X_train = np.array([2, 1, 1, 1, 1, 2, 2, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2])
        k = 7
        distance_measure = 3
        assert od_response_measure(q, x_new, X_train, k, distance_measure) == 1

    @staticmethod
    def test_63():
        """
        Case 1: Question 1: What's your gender?
                 Options: (1) Male. (2) Female.
                - One question with 2 options.
                - Arbitrary number of previous answers.
                - Arbitrary value of k.
                - Cosine Distance.
        The result should be 1 no matter what as one question means one axis, and
        one axis means that the angle between any two vectors is zero, and cos(0) = 1.
        """
        q = np.array([[2]])
        x_new = np.array([[2]])
        X_train = np.array([2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        k = 7
        distance_measure = 3
        assert od_response_measure(q, x_new, X_train, k, distance_measure) == 1

    @staticmethod
    def test_64():
        """
        Case 1: Question 1: What's your gender?
                 Options: (1) Male. (2) Female.
                - One question with 2 options.
                - Arbitrary number of previous answers.
                - Arbitrary value of k.
                - Cosine Distance.
        The result should be 1 no matter what as one question means one axis, and
        one axis means that the angle between any two vectors is zero, and cos(0) = 1.
        """
        q = np.array([[2]])
        x_new = np.array([[2]])
        X_train = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        k = 7
        distance_measure = 3
        assert od_response_measure(q, x_new, X_train, k, distance_measure) == 1

    @staticmethod
    def test_65():
        """
        Case 2: Question 1: What's your gender?
                 Options: (1) Male. (2) Female.
                Question 2: Rate Salah's pace.
                 Options: (1) Very slow. (2) Slow. (3) Normal. (4) Fast. (5) Very fast.
                - Two questions with arbitrary number of options.
                - Arbitrary number of previous answers.
                - Arbitrary value of k.
                - Euclidean Distance.
        The result should be 1 if and only if there exists k prev answers match the new
        one. 0 if and only if none of them match it.
        """
        q = np.array([[2],
                      [5]])
        x_new = np.array([[1],
                          [5]])
        X_train = np.array([[2, 1, 2, 1, 1, 2, 2, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2],
                           [5, 4, 5, 4, 3, 4, 5, 4, 3, 5, 4, 4, 5, 4, 5, 5, 4, 4]])
        k = 7
        distance_measure = 1

        delta = x_new - X_train
        total_distance = float(np.sum(np.sort(np.linalg.norm(delta, 2, axis=0))[:k]))
        max_distance = float(k * np.linalg.norm(q-1, 2, axis=0))

        assert od_response_measure(q, x_new, X_train, k, distance_measure) == 1 - (total_distance/max_distance)

    @staticmethod
    def test_66():
        """
        Case 2: Question 1: What's your gender?
                 Options: (1) Male. (2) Female.
                Question 2: Rate Salah's pace.
                 Options: (1) Very slow. (2) Slow. (3) Normal. (4) Fast. (5) Very fast.
                - Two questions with arbitrary number of options.
                - Arbitrary number of previous answers.
                - Arbitrary value of k.
                - Euclidean Distance.
        The result should be 1 if and only if there exists k prev answers match the new
        one. 0 if and only if none of them match it.
        """
        q = np.array([[2],
                      [5]])
        x_new = np.array([[1],
                          [1]])
        X_train = np.array([[2, 1, 2, 1, 1, 2, 2, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2],
                           [5, 4, 5, 4, 3, 4, 5, 4, 3, 5, 4, 4, 5, 4, 5, 5, 4, 4]])
        k = 9
        distance_measure = 1

        delta = x_new - X_train
        total_distance = float(np.sum(np.sort(np.linalg.norm(delta, 2, axis=0))[:k]))
        max_distance = float(k * np.linalg.norm(q-1, 2, axis=0))

        assert od_response_measure(q, x_new, X_train, k, distance_measure) == 1 - (total_distance/max_distance)

    @staticmethod
    def test_67():
        """
        Case 2: Question 1: What's your gender?
                 Options: (1) Male. (2) Female.
                Question 2: Rate Salah's pace.
                 Options: (1) Very slow. (2) Slow. (3) Normal. (4) Fast. (5) Very fast.
                - Two questions with arbitrary number of options.
                - Arbitrary number of previous answers.
                - Arbitrary value of k.
                - City-block Distance.
        The result should be 1 if and only if there exists k prev answers match the new
        one. 0 if and only if none of them match it.
        """
        q = np.array([[2],
                      [5]])
        x_new = np.array([[1],
                          [5]])
        X_train = np.array([[2, 1, 2, 1, 1, 2, 2, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2],
                           [5, 4, 5, 4, 3, 4, 5, 4, 3, 5, 4, 4, 5, 4, 5, 5, 4, 4]])
        k = 9
        distance_measure = 2

        delta = x_new - X_train
        distances = np.sum(np.abs(delta), axis=0)
        total_distance = float(np.sum(np.sort(distances)[:k]))
        max_distance = float(k * np.sum(q-1, axis=0))

        assert od_response_measure(q, x_new, X_train, k, distance_measure) == 1 - (total_distance/max_distance)

    @staticmethod
    def test_68():
        """
        Case 2: Question 1: What's your gender?
                 Options: (1) Male. (2) Female.
                Question 2: Rate Salah's pace.
                 Options: (1) Very slow. (2) Slow. (3) Normal. (4) Fast. (5) Very fast.
                - Two questions with arbitrary number of options.
                - Arbitrary number of previous answers.
                - Arbitrary value of k.
                - City-block Distance.
        The result should be 1 if and only if there exists k prev answers match the new
        one. 0 if and only if none of them match it.
     """
        q = np.array([[2],
                      [5]])
        x_new = np.array([[1],
                          [1]])
        X_train = np.array([[2, 1, 2, 1, 1, 2, 2, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2],
                           [5, 4, 5, 4, 3, 4, 5, 4, 3, 5, 4, 4, 5, 4, 5, 5, 4, 4]])
        k = 9
        distance_measure = 2

        delta = x_new - X_train
        distances = np.sum(np.abs(delta), axis=0)
        total_distance = float(np.sum(np.sort(distances)[:k]))
        max_distance = float(k * np.sum(q-1, axis=0))

        assert od_response_measure(q, x_new, X_train, k, distance_measure) == 1 - (total_distance/max_distance)

    @staticmethod
    def test_69():
        """
        Case 2: Question 1: What's your gender?
                 Options: (1) Male. (2) Female.
                Question 2: Rate Salah's pace.
                 Options: (1) Very slow. (2) Slow. (3) Normal. (4) Fast. (5) Very fast.
                - Two questions with arbitrary number of options.
                - Arbitrary number of previous answers.
                - Arbitrary value of k.
                - Cosine Distance.
        The result is 1 if and only if the angle between k previous answers and
        the new one is 0, and is 0 if and only if the angle between all of them
        and it is maximum possible angle in the space.
        """
        q = np.array([[2],
                      [5]])
        x_new = np.array([[1],
                          [5]])
        X_train = np.array([[2, 1, 2, 1, 1, 2, 2, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2],
                           [5, 4, 5, 4, 3, 4, 5, 4, 3, 5, 4, 4, 5, 4, 5, 5, 4, 4]])
        k = 9
        distance_measure = 3

        distances = 1 - np.dot(x_new.T, X_train) / (np.linalg.norm(x_new, 2) * np.linalg.norm(X_train, 2, axis=0))
        total_distance = float(np.sum(np.sort(distances)[:, :k]))
        max_distance = float(k * compute_cosine_max_distance_brute_force(q))

        assert od_response_measure(q, x_new, X_train, k, distance_measure) == pytest.approx(1 - (total_distance/max_distance))

    @staticmethod
    def test_70():
        """
        Case 2: Question 1: What's your gender?
                 Options: (1) Male. (2) Female.
                Question 2: Rate Salah's pace.
                 Options: (1) Very slow. (2) Slow. (3) Normal. (4) Fast. (5) Very fast.
                - Two questions with arbitrary number of options.
                - Arbitrary number of previous answers.
                - Arbitrary value of k.
                - Cosine Distance.
        The result is 1 if and only if the angle between k previous answers and
        the new one is 0, and is 0 if and only if the angle between all of them
        and it is (bi/2)
        """
        q = np.array([[2],
                      [5]])
        x_new = np.array([[1],
                          [1]])
        X_train = np.array([[2, 1, 2, 1, 1, 2, 2, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2],
                           [5, 4, 5, 4, 3, 4, 5, 4, 3, 5, 4, 4, 5, 4, 5, 5, 4, 4]])
        k = 9
        distance_measure = 3

        distances = 1 - np.dot(x_new.T, X_train) / (np.linalg.norm(x_new, 2) * np.linalg.norm(X_train, 2, axis=0))
        total_distance = float(np.sum(np.sort(distances)[:, :k]))
        max_distance = float(k * compute_cosine_max_distance_brute_force(q))

        assert od_response_measure(q, x_new, X_train, k, distance_measure) == pytest.approx(1 - (total_distance/max_distance))

    @staticmethod
    def test_71():
        """
        Trivial case 4: - Arbitrary number of questions with arbitrary number of answers
                        - No previous answers.
        For aribtrary valid value of k and any distance measure, result must be 1,
        as there is no information yet.
        """
        q = np.array([[3],
                      [5],
                      [5],
                      [2],
                      [7],
                      [3],
                      [2],
                      [4]])
        x_new = np.array([[1],
                          [3],
                          [5],
                          [1],
                          [3],
                          [2],
                          [2],
                          [3]])
        X_train = np.array([])
        k = 124
        distance_measure = 1
        assert od_response_measure(q, x_new, X_train, k, distance_measure) == 1

    @staticmethod
    def test_72():
        """
        Trivial case 4: - Arbitrary number of questions with arbitrary number of answers
                        - No previous answers.
        For aribtrary valid value of k and any distance measure, result must be 1,
        as there is no information yet.
        """
        q = np.array([[3],
                      [5],
                      [5],
                      [2],
                      [7],
                      [3],
                      [2],
                      [4]])
        x_new = np.array([[1],
                          [3],
                          [5],
                          [1],
                          [3],
                          [2],
                          [2],
                          [3]])
        X_train = np.array([])
        k = 124
        distance_measure = 2
        assert od_response_measure(q, x_new, X_train, k, distance_measure) == 1

    @staticmethod
    def test_73():
        """
        Trivial case 4: - Arbitrary number of questions with arbitrary number of answers
                        - No previous answers.
        For aribtrary valid value of k and any distance measure, result must be 1,
        as there is no information yet.
        """
        q = np.array([[3],
                      [5],
                      [5],
                      [2],
                      [7],
                      [3],
                      [2],
                      [4]])
        x_new = np.array([[1],
                          [3],
                          [5],
                          [1],
                          [3],
                          [2],
                          [2],
                          [3]])
        X_train = np.array([])
        k = 124
        distance_measure = 3
        assert od_response_measure(q, x_new, X_train, k, distance_measure) == 1

    @staticmethod
    def test_74():
        """
        Case 3: Question 1: What's your gender?
                 Options: (1) Male. (2) Female.
                Question 2: Rate Salah's pace.
                 Options: (1) Very slow. (2) Slow. (3) Normal. (4) Fast. (5) Very fast.
                Question 3: Rate Salah's defensive awareness.
                 Options: (1) Unaware. (2) Moderate (3) Aware.
                Question 4: Do you think Salah could play as a goalkeeper?
                 Options: (1) Yes. (2) No.
                - Arbitrary number of questions with arbitrary numbers of answers
                - Arbitrary number of previous answers.
                - Arbitrary value of k.
                - Euclidean Distance.
        The result should be 1 if and only if there exist k prev answers match the new
        one. 1/k if one of them match it. l/k if l of them match it. 0 if none of them match it.
        """
        q = np.array([[2],
                      [5],
                      [3],
                      [2]])
        x_new = np.array([[1],
                          [5],
                          [2],
                          [2]])
        X_train = np.array([[2, 1, 2, 1, 1, 2, 2, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2],
                            [5, 4, 5, 4, 3, 4, 5, 4, 3, 5, 4, 4, 5, 4, 5, 5, 4, 4],
                            [3, 2, 3, 2, 1, 2, 3, 2, 1, 2, 1, 2, 1, 3, 2, 1, 1, 3],
                            [1, 2, 1, 2, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2]])
        k = 5
        distance_measure = 1

        delta = x_new - X_train
        distances = np.linalg.norm(delta, 2, axis=0)
        total_distance = float(np.sum(np.sort(distances)[:k]))
        max_distance = float(k * np.linalg.norm(q-1, 2, axis=0))

        assert od_response_measure(q, x_new, X_train, k, distance_measure) == 1 - (total_distance/max_distance)

    @staticmethod
    def test_75():
        """
        Case 3: Question 1: What's your gender?
                 Options: (1) Male. (2) Female.
                Question 2: Rate Salah's pace.
                 Options: (1) Very slow. (2) Slow. (3) Normal. (4) Fast. (5) Very fast.
                Question 3: Rate Salah's defensive awareness.
                 Options: (1) Unaware. (2) Moderate (3) Aware.
                Question 4: Do you think Salah could play as a goalkeeper?
                 Options: (1) Yes. (2) No.
                - Arbitrary number of questions with arbitrary numbers of answers
                - Arbitrary number of previous answers.
                - Arbitrary value of k.
                - Euclidean Distance.
        The result should be 1 if and only if there exist k prev answers match the new
        one. 1/k if one of them match it. l/k if l of them match it. 0 if none of them match it.
        """
        q = np.array([[2],
                      [5],
                      [3],
                      [2]])
        x_new = np.array([[1],
                          [1],
                          [1],
                          [1]])
        X_train = np.array([[2, 1, 2, 1, 1, 2, 2, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2],
                            [5, 4, 5, 4, 3, 4, 5, 4, 3, 5, 4, 4, 5, 4, 5, 5, 4, 4],
                            [3, 2, 3, 2, 1, 2, 3, 2, 1, 2, 1, 2, 1, 3, 2, 1, 1, 3],
                            [1, 2, 1, 2, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2]])
        k = 5
        distance_measure = 1

        delta = x_new - X_train
        distances = np.linalg.norm(delta, 2, axis=0)
        total_distance = float(np.sum(np.sort(distances)[:k]))
        max_distance = float(k * np.linalg.norm(q-1, 2, axis=0))

        assert od_response_measure(q, x_new, X_train, k, distance_measure) == 1 - (total_distance/max_distance)

    @staticmethod
    def test_76():
        """
        Case 3: Question 1: What's your gender?
                 Options: (1) Male. (2) Female.
                Question 2: Rate Salah's pace.
                 Options: (1) Very slow. (2) Slow. (3) Normal. (4) Fast. (5) Very fast.
                Question 3: Rate Salah's defensive awareness.
                 Options: (1) Unaware. (2) Moderate (3) Aware.
                Question 4: Do you think Salah could play as a goalkeeper?
                 Options: (1) Yes. (2) No.
                - Arbitrary number of questions with arbitrary numbers of answers
                - Arbitrary number of previous answers.
                - Arbitrary value of k.
                - City-block Distance.
        The result should be 1 if and only if there exist k prev answers match the new
        one. 1/k if one of them match it. l/k if l of them match it. 0 if none of them match it.
        """
        q = np.array([[2],
                      [5],
                      [3],
                      [2]])
        x_new = np.array([[1],
                          [5],
                          [2],
                          [2]])
        X_train = np.array([[2, 1, 2, 1, 1, 2, 2, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2],
                            [5, 4, 5, 4, 3, 4, 5, 4, 3, 5, 4, 4, 5, 4, 5, 5, 4, 4],
                            [3, 2, 3, 2, 1, 2, 3, 2, 1, 2, 1, 2, 1, 3, 2, 1, 1, 3],
                            [1, 2, 1, 2, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2]])
        k = 4
        distance_measure = 2

        delta = x_new - X_train
        distances = np.sum(np.abs(delta), axis=0)
        total_distance = float(np.sum(np.sort(distances)[:k]))
        max_distance = float(k * np.sum(q-1, axis=0))

        assert od_response_measure(q, x_new, X_train, k, distance_measure) == 1 - (total_distance/max_distance)

    @staticmethod
    def test_77():
        """
        Case 3: Question 1: What's your gender?
                 Options: (1) Male. (2) Female.
                Question 2: Rate Salah's pace.
                 Options: (1) Very slow. (2) Slow. (3) Normal. (4) Fast. (5) Very fast.
                Question 3: Rate Salah's defensive awareness.
                 Options: (1) Unaware. (2) Moderate (3) Aware.
                Question 4: Do you think Salah could play as a goalkeeper?
                 Options: (1) Yes. (2) No.
                - Arbitrary number of questions with arbitrary numbers of answers
                - Arbitrary number of previous answers.
                - Arbitrary value of k.
                - City-block Distance.
        The result should be 1 if and only if there exist k prev answers match the new
        one. 1/k if one of them match it. l/k if l of them match it. 0 if none of them match it.
        """
        q = np.array([[2],
                      [5],
                      [3],
                      [2]])
        x_new = np.array([[1],
                          [1],
                          [1],
                          [1]])
        X_train = np.array([[2, 1, 2, 1, 1, 2, 2, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2],
                            [5, 4, 5, 4, 3, 4, 5, 4, 3, 5, 4, 4, 5, 4, 5, 5, 4, 4],
                            [3, 2, 3, 2, 1, 2, 3, 2, 1, 2, 1, 2, 1, 3, 2, 1, 1, 3],
                            [1, 2, 1, 2, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2]])
        k = 4
        distance_measure = 2

        delta = x_new - X_train
        distances = np.sum(np.abs(delta), axis=0)
        total_distance = float(np.sum(np.sort(distances)[:k]))
        max_distance = float(k * np.sum(q-1, axis=0))

        assert od_response_measure(q, x_new, X_train, k, distance_measure) == 1 - (total_distance/max_distance)

    @staticmethod
    def test_78():
        """
        Case 3: Question 1: What's your gender?
                 Options: (1) Male. (2) Female.
                Question 2: Rate Salah's pace.
                 Options: (1) Very slow. (2) Slow. (3) Normal. (4) Fast. (5) Very fast.
                Question 3: Rate Salah's defensive awareness.
                 Options: (1) Unaware. (2) Moderate (3) Aware.
                Question 4: Do you think Salah could play as a goalkeeper?
                 Options: (1) Yes. (2) No.
                - Arbitrary number of questions with arbitrary numbers of answers
                - Arbitrary number of previous answers.
                - Arbitrary value of k.
                - Cosine Distance.
        The result is 1 if and only if the angle between k previous answers and
        the new one is 0, and is 0 if and only if the angle between all of them
        and it is (bi/2)
        """
        q = np.array([[2],
                      [5],
                      [3],
                      [2]])
        x_new = np.array([[1],
                          [5],
                          [2],
                          [2]])
        X_train = np.array([[2, 1, 2, 1, 1, 2, 2, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2],
                            [5, 4, 5, 4, 3, 4, 5, 4, 3, 5, 4, 4, 5, 4, 5, 5, 4, 4],
                            [3, 2, 3, 2, 1, 2, 3, 2, 1, 2, 1, 2, 1, 3, 2, 1, 1, 3],
                            [1, 2, 1, 2, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2]])
        k = 4
        distance_measure = 3

        distances = 1 - np.dot(x_new.T, X_train) / (np.linalg.norm(x_new, 2) * np.linalg.norm(X_train, 2, axis=0))
        total_distance = float(np.sum(np.sort(distances)[:, :k]))
        max_distance = float(k * compute_cosine_max_distance_brute_force(q))

        assert od_response_measure(q, x_new, X_train, k, distance_measure) == pytest.approx(1 - (total_distance/max_distance))

    @staticmethod
    def test_79():
        """
        Case 3: Question 1: What's your gender?
                 Options: (1) Male. (2) Female.
                Question 2: Rate Salah's pace.
                 Options: (1) Very slow. (2) Slow. (3) Normal. (4) Fast. (5) Very fast.
                Question 3: Rate Salah's defensive awareness.
                 Options: (1) Unaware. (2) Moderate (3) Aware.
                Question 4: Do you think Salah could play as a goalkeeper?
                 Options: (1) Yes. (2) No.
                - Arbitrary number of questions with arbitrary numbers of answers
                - Arbitrary number of previous answers.
                - Arbitrary value of k.
                - Cosine Distance.
        The result is 1 if and only if the angle between k previous answers and
        the new one is 0, and is 0 if and only if the angle between all of them
        and it is (bi/2)
        """
        q = np.array([[2],
                      [5],
                      [3],
                      [2]])
        x_new = np.array([[1],
                          [1],
                          [1],
                          [1]])
        X_train = np.array([[2, 1, 2, 1, 1, 2, 2, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2],
                            [5, 4, 5, 4, 3, 4, 5, 4, 3, 5, 4, 4, 5, 4, 5, 5, 4, 4],
                            [3, 2, 3, 2, 1, 2, 3, 2, 1, 2, 1, 2, 1, 3, 2, 1, 1, 3],
                            [1, 2, 1, 2, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2]])
        k = 4
        distance_measure = 3

        distances = 1 - np.dot(x_new.T, X_train) / (np.linalg.norm(x_new, 2) * np.linalg.norm(X_train, 2, axis=0))
        total_distance = float(np.sum(np.sort(distances)[:, :k]))
        max_distance = float(k * compute_cosine_max_distance_brute_force(q))

        assert od_response_measure(q, x_new, X_train, k, distance_measure) == pytest.approx(1 - (total_distance/max_distance))
