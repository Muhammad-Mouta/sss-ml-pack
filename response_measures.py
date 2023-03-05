import warnings

import numpy as np

from .utils.response_measures import ODResponseMeasure as odu
from .utils.response_measures import MultimodalDistroModel as mdmu
from .utils.response_measures import DependencyMap as dmu


class MultimodalDistroModel:
    def __init__(self, q=None, R=None, m=None, s=None):
        """
        Initiates a Multimodal Distro Model instance.

        Parameters
        ----------
            q : numpy.array, shape = (n, 1) or (1, n) or (n,), dtype=int, default : None.
                A vector containing the number of answers for each question, where
                (theta_i) is the number of answers for the (i_th) question.
            R : numpy.array, dtype=float, shape = (n, max(q)), default : None.
                It is one part of the linear combination that forms the distro (1/gamma) * (R @ v).
            m : numpy.array, dtype=float, shape = (n, 1), default : None.
                The number of responses so far.
            s : numpy.array, dtype=float, default : None.
                The standard deviation of the distro at each modal. It controls
                the width of the distro at each model. In other words, if an
                answer is chosen, it affects how much the probability densities of
                nearby answers are affected.

        Notes
        -----
            - Rgarding s:
                - If s is not given, it is computed as l/4, where l is the number
                of possible answers for the question. If a value of an element
                within s is None, it is computed as l/4.
                - If the value of an element within s is equal to zero, the
                answers of the corresponding question are considered discrete,
                which means that choosing an answer doesn't affect the
                probability density of any of the nearby answers.

            - A model must be initialized by one of 2 methods:
                1- Providing q:
                    - In this case, R, m are infered from q. s is l/4, and the
                      model is considered continuous.
                    - Even if R and m are given, they are ignored.
                    - However, if s is given, it is not infered.
                2- Providing R:
                    - Providing m is optional, if it is not provided, it is infered from R.
                    - If s is not given it is l/4, and the model is considered
                      continuous.
                    - If s is given, the model is completely defined and nothing
                      is infered.
        """
        _q, _R, _m, _s, _ = mdmu.handle_input(q=q, R=R, m=m, s=s)

        # If q is provided
        if type(_q) != type(None):
            self.R, self.m, self.s = self._create_distro(_q, _s)

        # If R is provided
        elif type(_R) != type(None):
            # If m is not provided
            if type(_m) == type(None):
                _m = mdmu.compute_m(_R)
            self.R, self.m, self.s = _R, _m, mdmu.compute_s(_s, _R)

        # Otherwise, raise an error
        else:
            raise ValueError("Wrong Initialization, check the documentation for more information.")


    @staticmethod
    def _create_distro(q, s=None):
        """
        Creates a distro by initiating its parameters.

        Parameters
        ----------
            q : numpy.array, shape = (n, 1) or (1, n) or (n,), dtype=int64s.
                A vector containing the number of answers for each question, where
                (theta_i) is the number of answers for the (i_th) question.
            s : numpy.array, dtype=float64, default : None.
                The standard deviation of the distro at each modal. It controls
                the width of the distro at each model. In other words, if an
                answer is chosen, it affects how much the probability densities of
                nearby answers are affected.

        Returns
        -------
            numpy.array, shape = (n, argmax(q)), which represents R, which
                is one part of the linear combination that forms the distro (1/gamma) * (R @ v).
            int64, which represents m, the number of responses so far.
            float64, which represents s. If s is given, it is returned
                unchanged. If s is not given, it is computed as l/4, where l is the number
                of possible answers for the question. If a value of an element within s is None,
                it is computed as l/4. If the value of an element within s is equal to zero, the
                answers of the corresponding question are considered discrete,
                which means that choosing an answer doesn't affect the
                probability density of any of the nearby answers.

        Notes
        -----
            - Rgarding s:
                - If s is not given or the value of an element within it is None,
                it is computed as l/4.
                - If the value of an element within s is equal to zero, the
                answers of the corresponding question are considered discrete,
                which means that choosing an answer doesn't affect the
                probability density of any of the nearby answers.
        """
        # Compute R
        R = np.zeros((q.shape[0], np.max(q))).astype("int64")
        for i in range(R.shape[0]):
            R[i, :q[i, 0]] = 1

        # Compute m
        m = q.copy()

        return R, m, mdmu.compute_s(s, R)


    def update_distro(self, x):
        """
        Updates the parameters of the model (R, m).

        Parameters
        ----------
            x : numpy.array, shape = (n, 1).
                A vector containing the new answer which is to be measured, where
                (chi_i) is the answer to the (i_th) question.

        Returns
        -------
            numpy.array, shape = (n, argmax(q)), which represents R, which
                is one part of the linear combination that forms the distro (1/gamma) * (R @ v).
            int64, which represents m, the number of responses so far.
        """
        _, _, _, _, _x = mdmu.handle_input(x=x)

        # Do without any question without a response
        row_mask = _x[:, 0] != 0
        _x = _x[row_mask, :]
        _R = self.R[row_mask, :]
        _m = self.m[row_mask, :]

        # Update the parameters using the new response
        for i in range(_R.shape[0]):
            _R[i, _x[i, :] - 1] += 1
        _m += 1

        # Update the models' self parameters
        self.R[row_mask, :] = _R
        self.m[row_mask, :] = _m

        return self.R, self.m


    def compute_score(self, x):
        """
        Computes the reliability score of the given response.

        Parameters
        ----------
            x : numpy.array, shape = (n, 1).
                A vector containing the new answer which is to be measured, where
                (chi_i) is the answer to the (i_th) question.

        Returns
        -------
            float64, [0, 1], which represents the reliability score of the response.
        """
        _, _, _, _, _x = mdmu.handle_input(x=x)

        # Do without any question without a response
        row_mask = _x[:, 0] != 0
        _x = _x[row_mask, :]
        _R = self.R[row_mask, :]
        _m = self.m[row_mask, :]
        _s = self.s[row_mask, :]

        # Compute the probability density of the response
        p = mdmu.compute_prob_density(_x, _R, _m, _s)

        # Get the response with the most answer count
        max_x = np.argmax(_R, axis=1).reshape(-1, 1) + 1
        # Roughly, compute the highest acquirable probability density
        p_max = mdmu.compute_prob_density(max_x, _R, _m, _s)

        # Get the percentage of p/p_max
        score = float(np.sum(p) / np.sum(p_max))

        # If the score exceeds 1, return 1
        return score if score < 1 else 1


    def auto_answer(self, y=None, neg_displacement=None, pos_displacement=None, return_probs=False):
        """
        Automatically picks answers for the specified questions based on the
        distro.

        Parameters
        ----------
            y : numpy.array, dtype=boolean, shape=(n, 1) or (1, n) or (n, ), default=None.
                A boolean vector to indicate which question to auto-answer.
                If (psi_i) == 1, the corresponding question will be auto answered,
                else, it won't.
                If not given, it is assumed that all questions will be auto-answered.
            neg_displacement: np.array, dtype=float, shape=(n, 1) or (1, n) or (n, ).
                The displacement in the negative direction for integrating.
                If not given, it is assumed to be 0.5 for all answers.
            pos_displacement: np.array, dtype=float, shape=(n, 1) or (1, n) or (n, ).
                The displacement in the positive direction for integrating.
                If not given, it is assumed to be 0.5 for all answers.
            return_probs: bool.
                If True, The function also returns a vector containing the probabilities
                of each of the returned answers.

        Returns
        -------
            np.array, shape=(n, 1) : It contains the answers, where (chi_i) is
            the answer of the (i_th) question. The corresponding values of the
            questions that aren't answered are all set to zeros.
            np.array, shape=(n, 1) Optional : It contains the probability of each
            of the returned answer.
        """
        # Handle the input
        _y, _neg_displacement, _pos_displacement = mdmu.handle_input_auto_answer(y=y, neg_displacement=neg_displacement, pos_displacement=pos_displacement)

        if type(_y) == type(None):
            _y = np.ones((self.R.shape[0], 1))
        if type(_neg_displacement) == type(None):
            _neg_displacement = np.ones(_y.shape) * 0.5
        if type(_pos_displacement) == type(None):
            _pos_displacement = np.ones(_y.shape) * 0.5

        # Initiate the response vector and the probs vector
        x = np.zeros(_y.shape)
        probs = np.zeros(_y.shape)

        # Do without the questions that won't be auto answered
        row_mask = _y[:, 0] != 0
        _R = self.R[row_mask, :]
        _m = self.m[row_mask, :]
        _s = self.s[row_mask, :]

        # Initiate the p matrix, which contains the discrete probability distributions
        # for each answer. The i_th row corresponds the i_th question and
        # the j_th column corresponds to the j_th answer. The last column has
        # complement probabilities, that is values that makes the probabilities
        # in each row sum to 1.
        p = np.zeros((_R.shape[0], _R.shape[1]+1))

        # Get the probability for each answer
        for i in range(p.shape[1] - 1):
            p[:, i] = mdmu.compute_prob((np.ones((_R.shape[0], 1)) * (i+1)), _R, _m, _s, _neg_displacement, _pos_displacement).reshape(-1, )

        # Set the probabilities of the no answers to zeros
        mask = np.hstack((_R == 0, np.zeros((_R.shape[0], 1)))).astype("bool")
        p[mask] = 0

        # Get the values of the last row
        sums = np.sum(p[:, :-1], axis=1)
        p[:, -1] = 1 - sums

        # Pick the answers and save them to x and save their probabilities in probs
        choices = np.arange(_R.shape[1]+1) + 1
        choices[-1] = 0
        answers = np.zeros((_R.shape[0], ))
        probabilities = np.zeros(answers.shape)
        for i in range(_R.shape[0]):
            while(True):
                answer = np.random.choice(choices, p=p[i, :])
                if answer != 0:
                    answers[i] = answer
                    probabilities[i] = p[i, answer-1]
                    break
        x[_y.astype("bool")] = answers
        probs[_y.astype("bool")] = probabilities

        return (x, probs) if return_probs else x


class DependencyMap:
    def __init__(self, M=None, shape=None):
        """
        Initiates a Dependency Map.

        Parameters
        ----------
            M : numpy.array, dtype=float, shape = shape, default : None.
                The Dependency Map numpy array.
            shape : tuple of length 2, dtype=int, default : None.
                The length and the width of the map, the first entry in the tuple
                is axis_0 (y_axis), and the second entry is axis_1 (x_axis).

        Notes
        -----
            - A map must be initialized by one of 2 methods:
                1- Providing M:
                    - In this case, the map is the given numpy.array.
                2- Providing shape:
                    - In this case a numpy array of zeros is initialized with
                    the given shape.
            - If M is provided, shape is ignored.
        """
        _M, _shape = dmu.handle_input(M, shape)

        # If M is provided
        if type(_M) != type(None):
            self.M = _M

        # If shape is provided
        elif type(_shape) != type(None):
            self.M = np.zeros(shape)
            self.shape = _shape

        # Otherwise, raise an error
        else:
            raise ValueError("Wrong Initialization, check the documentation for more information.")


    def update(self, q_0, q_1):
        """
        Updates the map given two vectors, the axis_0 vector and the axis_1 vector.

        Parameters
        ----------
            q_0: numpy.array, shape=(self.shape[0], 1).
                The vector corresponding to axis_0.
            q_1: numpy.array, shape=(1, self.shape[1]).
                The vector corresponding to axis_1.

        Returns
        -------
            numpy.array, the resultant map.

        """
        _q_0, _q_1 = dmu.handle_input_update(q_0, q_1)
        self.M = self.M + (_q_0 == _q_1)
        return self.M


    def M_to_R(self, q_1, R_width, m):
        """
        Transforms the map M into parameters R.

        Parameters
        ----------
            q_1: numpy.array, shape=(1, self.shape[1]).
                The vector corresponding to axis_1.
            R_width: int.
                The width of the R matrix.
            m : numpy.array, dtype=float, shape = (shape[0], 1).
                The number of responses so far.

        Returns
        -------
            numpy.array, R.

        Notes
        -----
            q_1 is one-based.
        """
        _q_1, _R_width, _m = dmu.handle_input_M_to_R(q_1, R_width, m)
        _M = self.M

        # Obtain the mask, keep entries with max value and any entry whose value is >= 0.5
        mask = ((_M == _M.max(axis=1).reshape(-1, 1)) | (_M >= (0.5 * _m)))
        _M[~mask] = 0


        # Initialize R
        R = np.zeros((_M.shape[0], _R_width))

        # for i in range(self.M.shape[0]):
        #     for j in range(self.M.shape[1]):
        #         R[i, q_1[j]] += M[i, j]

        for i in range(self.M.shape[1]):
            R[:, int(_q_1[0, i]-1)] = np.maximum(R[:, int(_q_1[0, i]-1)], _M[:, i])

        return R


def time_response_measure(respondent_time, SURVEY_READING_TIME):
    """
    Returns a score measuring how acceptable the response is according to
    the time taken by the respondent to answer the entire survey.

    Parameters
    ----------
        respondent_time : float, must be greater than or equal to 0.
            The actual time spent by the user responding to the survey.

        SURVEY_READING_TIME : float, must be greater than 0.
            The expected time spent by the user reading the survey.

    Returns
    -------
        float in the range [0, 1], where 0 means completely unacceptable and
        1 means completely acceptable

    Notes
    -----
        The unit of time could be anything as long as it is the same for both
        parameters. (e.g. if respondent_time is in minutes, SURVEY_READING_TIME
        must also be in minutes)
    """
    if (respondent_time < 0) or (SURVEY_READING_TIME <= 0):
        raise ValueError("respondent_time must be greater than or equal to 0, and SURVEY_READING_TIME must be greater than 0.")

    if respondent_time < SURVEY_READING_TIME:
        return float(respondent_time/SURVEY_READING_TIME)
    else:
        return 1


def od_response_measure(q, x_new, X_train, k=3, distance_measure=1, zero_based=False, print_q=False):
    """
    Returns a score measuring how acceptable the response is according to
    previous responses using outlier detection.

    Parameters
    ----------
        q : numpy.array, shape = (n, 1) or (1, n) or (n,).
            A vector containing the number of answers for each question, where
            (theta_i) is the number of answers for the (i_th) question.
        x_new : numpy.array, shape = (n, 1).
            A vector containing the new answer which is to be measured, where
            (chi_i) is the answer to the (i_th) question.
        X_train : numpy.array, shape = (n, m).
            A matrix containing the previous answers, where (chi_i,j) is the
            answer to the (i_th) question of the (j_th) respondent.
        k : int, greater than zero, default: 3.
            The number of neighbours.
        distance_measure : int, default: 1.
            The distance measure used to meaesure distance between neighbours.
            It is one of:
                (1): Euclidean.
                (2): City Block.
                (3): Cosine.
        zero_based : boolean, default: False.
            Set it True if the indices of your answers begin with 0, and
            set it False if they begin with 1.
            (i.e. if your first answer is indexed 0, set it True)

    Returns
    -------
        float in the range [0, 1], where 0 means completely unacceptable and
        1 means completely acceptable.
    """
    # ---------------------------- Handle errors ---------------------------- #

    # ------- Self-value Errors:
    allowed_distance_measure_values = [1, 2, 3]
    if (q.size == 0):
        raise ValueError("The q vector can't be empty")
    # -------
    if (x_new.size == 0):
        raise ValueError("The x_new vector can't be empty")
    # -------
    if (k < 0):
        raise ValueError(f"k must be greater than or equal to zero, found {k}.")
    # -------
    if not (distance_measure in allowed_distance_measure_values):
        raise ValueError("Wrong distance_measure value, check the allowed values in the documentation")

    # ------- Self-dimension Errors:
    if q.ndim == 1:
        n = 1
    elif q.ndim == 2:
        if min(q.shape) != 1:
            raise ValueError(f"q must be a vector of size (n, 1) or (1, n) or (n,), found {q.shape}")
        else:
            n = max(q.shape)
    else:
        raise ValueError(f"q must be a vector of size (n, 1) or (1, n) or (n,), found {q.shape}")

    # Make a copy of q.
    q = q.copy()
    # Reshape q to make sure it is a column vector
    q = q.reshape(n, 1)

    # ------- Cross-dimension Errors:
    # Make a copy of x_new.
    x_new = x_new.copy()
    # Reshape x_new to make sure it is a column vector
    try:
        x_new = x_new.reshape(n, 1)
    except ValueError:
        raise ValueError(f"x_new and q must have the same dimensions, found q.shape = {q.shape}, x_new.shape = {x_new.shape}")
    # -------
    # Make a copy of X_train.
    X_train = X_train.copy()

    if X_train.ndim == 1:
        # Reshape X_train to make sure its shape is (n, m)
        if X_train.size != 0:
            X_train = X_train.reshape(1, X_train.shape[0])
        else:
            X_train = X_train.reshape(n, X_train.shape[0])
    if X_train.shape[0] != n:
        raise ValueError(f"X_train and q must have the same first dimension but found q.shape = {q.shape}, X_train.shape = {X_train.shape}")
    m = X_train.shape[1]

    # ------- Cross-value Errors
    for i in range(n):
        if q[i, 0] < 2:
            if q[i, 0] == 1:
                warnings.warn(f"Question {i} has only one option")
            else:
                raise ValueError(f"A question must have at least one option")

        if zero_based:
            if x_new[i, 0] >= q[i, 0] or x_new[i, 0] < 0:
                raise ValueError(f"Question {i} answer is out of range, found {x_new[i, 0]}; allowed range = [0, {q[i, 0]}]")
        else:
            if x_new[i, 0] > q[i, 0] or x_new[i, 0] <= 0:
                raise ValueError(f"Question {i} answer is out of range, found {x_new[i, 0]}; allowed range = [0, {q[i, 0]}]")

        for j in range(m):
            if zero_based:
                if X_train[i, j] >= q[i, 0] or X_train[i, j] < 0:
                    raise ValueError(f"Question {i} answer of the {j}_th respondent is out of range, found {X_train[i, j]}; allowed range = [0, {q[i, 0]}]")
            else:
                if X_train[i, j] > q[i, 0] or X_train[i, j] <= 0:
                    raise ValueError(f"Question {i} answer of the {j}_th respondent is out of range, found {X_train[i, j]}; allowed range = [0, {q[i, 0]}]")

    # ----------------------------------------------------------------------- #

    # Make sure k doesn't exceed m.
    k = min(k, m)

    # If X_train is empty return 1
    if X_train.size == 0:
        return 1

    # Compute the maximum distance between any two responses
    max_distance = k * odu.compute_max_distance(q, distance_measure)

    # If max_distance == 0, this means that only one response is allowed.
    if max_distance == 0:
        return 1

    # Compute the distances between the new response and previous responses
    distances = odu.compute_distances(x_new, X_train, distance_measure, zero_based)

    # Compute the cumulative distance between the new response and the k nearest responses
    cumulative_distance = odu.cumulate_distances(distances, k)

    # Return the od response measure
    return (1 - (cumulative_distance/max_distance))
