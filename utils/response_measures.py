from scipy.special import erf
from numpy.linalg import norm
from numpy import array, sum, abs, divide, zeros, ones, sort, hstack, sqrt, arange, exp, pi, isnan

from .utils import generate_2_corner_points


class MultimodalDistroModel:
    @staticmethod
    def handle_input(q=None, R=None, m=None, s=None, x=None):
        """
        Ensures that the input is in the desired form, and
        raises errors and warnings if necessary.

        Parameters
        ----------
            q : numpy.array, shape = (n, 1) or (1, n) or (n,), default : None.
                A vector containing the number of answers for each question, where
                (theta_i) is the number of answers for the (i_th) question.
            R : numpy.array, shape = (n, max(q)), default : None.
                It is one part of the linear combination that forms the distro (1/gamma) * (R @ v).
            m : numpy.array, shape = (n, 1), default : None.
                The number of responses so far.
            s : numpy.array, dtype=float64, default : None.
                The standard deviation of the distro at each modal. It controls
                the width of the distro at each model. In other words, if an
                answer is chosen, it affects how much the probability densities of
                nearby answers are affected.

        Returns
        -------
            The input in the desired form.
            q : numpy.array, shape = (n, 1), dtype=int64.
            R : numpy.array, shape = (n, max(q)), dtype=float64.
            m : numpy.array, shape = (n, 1), dtype=float64.
            s : numpy.array, shape = (n, 1), dtype=float64.
        """
        if type(q) != type(None):
            # Cast it to np.array
            _q = array(q)

            # ------- Self-dimension Errors:
            if _q.ndim == 1:
                n = _q.size
            elif _q.ndim == 2:
                if min(_q.shape) != 1:
                    raise ValueError(f"q must be a vector of size (n, 1) or (1, n) or (n,), found {q.shape}")
                else:
                    n = max(_q.shape)
            else:
                raise ValueError(f"q must be a vector of size (n, 1) or (1, n) or (n,), found {q.shape}")

            # Reshape q to make sure it is a column vector and cast it to int64
            _q = _q.reshape(n, 1).astype("int64")
        else:
            _q = None

        if type(R) != type(None):
            _R = array(R)

            # ------- Self-dimension Errors:
            if _R.ndim == 1:
                _R = _R.reshape(1, -1).astype("float64")
            else:
                _R = _R.astype("float64")
        else:
            _R = None

        if type(m) != type(None):
            _m = array(m)

            # ------- Self-dimension Errors:
            if _m.ndim == 1:
                n = _m.size
            elif _m.ndim == 2:
                if min(_m.shape) != 1:
                    raise ValueError(f"m must be a vector of size (n, 1) or (1, n) or (n,), found {m.shape}")
                else:
                    n = max(_m.shape)
            else:
                raise ValueError(f"m must be a vector of size (n, 1) or (1, n) or (n,), found {m.shape}")

            # Reshape m to make sure it is a column vector and cast it to float64
            _m = _m.reshape(n, 1).astype("float64")
        else:
            _m = None

        if type(s) != type(None):
            _s = array(s)

            # ------- Self-dimension Errors:
            if _s.ndim == 1:
                n = _s.size
            elif _s.ndim == 2:
                if min(_s.shape) != 1:
                    raise ValueError(f"s must be a vector of size (n, 1) or (1, n) or (n,), found {s.shape}")
                else:
                    n = max(_s.shape)
            else:
                raise ValueError(f"s must be a vector of size (n, 1) or (1, n) or (n,), found {s.shape}")

            # Reshape m to make sure it is a column vector and cast it to int64
            _s = _s.reshape(n, 1).astype("float64")
        else:
            _s = None


        if type(x) != type(None):
            _x = array(x)

            # ------- Self-dimension Errors:
            if _x.ndim == 1:
                n = _x.size
            elif _x.ndim == 2:
                if min(_x.shape) != 1:
                    raise ValueError(f"x must be a vector of size (n, 1) or (1, n) or (n,), found {x.shape}")
                else:
                    n = max(_x.shape)
            else:
                raise ValueError(f"x must be a vector of size (n, 1) or (1, n) or (n,), found {x.shape}")

            # Reshape x to make sure it is a column vector and cast it to int64
            _x = _x.reshape(n, 1).astype("int64")
        else:
            _x = None

        return _q, _R, _m, _s, _x


    def handle_input_auto_answer(y=None, neg_displacement=None, pos_displacement=None):
        """
        Ensures that the input is in the desired form, and
        raises errors and warnings if necessary.

        Parameters
        ----------
            y : numpy.array, dtype=boolean, shape=(n, 1) or (1, n) or (n, ).
                A boolean vector to indicate which question to auto-answer.
                If (psi_i) == 1, the corresponding question will be auto answered,
                else, it won't.
            neg_displacement: np.array, dtype=float, shape=(n, 1) or (1, n) or (n, ).
                The displacement in the negative direction for integrating.
            pos_displacement: np.array, dtype=float, shape=(n, 1) or (1, n) or (n, ).
                The displacement in the positive direction for integrating.

        Returns
        -------
            The input in the desired form.
            y : numpy.array, dtype=boolean, shape=(n, 1).
            neg_displacement: np.array, dtype=float, shape=(n, 1).
            pos_displacement: np.array, dtype=float, shape=(n, 1).
        """
        if type(y) != type(None):
            # Cast it to np.array
            _y = array(y)

            # ------- Self-dimension Errors:
            if _y.ndim == 1:
                n = _y.size
            elif _y.ndim == 2:
                if min(_y.shape) != 1:
                    raise ValueError(f"y must be a vector of size (n, 1) or (1, n) or (n,), found {y.shape}")
                else:
                    n = max(_y.shape)
            else:
                raise ValueError(f"y must be a vector of size (n, 1) or (1, n) or (n,), found {y.shape}")

            # Reshape y to make sure it is a column vector and cast it to bool
            _y = _y.reshape(n, 1).astype("bool")
        else:
            _y = None


        if type(neg_displacement) != type(None):
            # Cast it to np.array
            _neg_displacement = array(neg_displacement)

            # ------- Self-dimension Errors:
            if _neg_displacement.ndim == 1:
                n = _neg_displacement.size
            elif _neg_displacement.ndim == 2:
                if min(_neg_displacement.shape) != 1:
                    raise ValueError(f"neg_displacement must be a vector of size (n, 1) or (1, n) or (n,), found {neg_displacement.shape}")
                else:
                    n = max(_neg_displacement.shape)
            else:
                raise ValueError(f"neg_displacement must be a vector of size (n, 1) or (1, n) or (n,), found {neg_displacement.shape}")

            # Reshape neg_displacement to make sure it is a column vector and cast it to float64
            _neg_displacement = _neg_displacement.reshape(n, 1).astype("float64")
        else:
            _neg_displacement = None


        if type(pos_displacement) != type(None):
            # Cast it to np.array
            _pos_displacement = array(pos_displacement)

            # ------- Self-dimension Errors:
            if _pos_displacement.ndim == 1:
                n = _pos_displacement.size
            elif _pos_displacement.ndim == 2:
                if min(_pos_displacement.shape) != 1:
                    raise ValueError(f"pos_displacement must be a vector of size (n, 1) or (1, n) or (n,), found {pos_displacement.shape}")
                else:
                    n = max(_pos_displacement.shape)
            else:
                raise ValueError(f"pos_displacement must be a vector of size (n, 1) or (1, n) or (n,), found {pos_displacement.shape}")

            # Reshape pos_displacement to make sure it is a column vector and cast it to float64
            _pos_displacement = _pos_displacement.reshape(n, 1).astype("float64")
        else:
            _pos_displacement = None

        return _y, _neg_displacement, _pos_displacement

    @staticmethod
    def compute_m(R):
        """
        Infers m from R.

        Parameters
        ----------
            R : numpy.array, shape = (n, argmax(q)), default : None.
                It is one part of the linear combination that forms the distro (1/gamma) * (R @ v).

        Returns
        -------
            np.array, shape = (R.shape[0], 1), dtype=float64, which represents
            the number of responses so far for each question.
        """
        return sum(R, axis=1).reshape(-1, 1)


    @staticmethod
    def compute_s(s, R):
        """
        Computes the s.

        Parameters
        ----------
            s : numpy.array, dtype=float64, >=0.
                The standard deviation of the distro at each modal. It controls
                the width of the distro at each model. In other words, if an
                answer is chosen, it affects how much the probability densities of
                nearby answers are affected.
            R : numpy.array, dtype=float, shape = (n, max(q)), default : None.
                It is one part of the linear combination that forms the distro (1/gamma) * (R @ v).

        Returns
        -------
            float64, which represents s. If s is given, it is returned
            unchanged. If s is None, it is computed as l/4, where l is the number
            of possible answers for the question. If a value of an element within s is None,
            it is computed as l/4. If the value of an element within s is zero, the
            answers of the corresponding question are considered discrete,
            which means that choosing an answer doesn't affect the
            probability density of any of the nearby answers.
        """
        l = sum(R != 0, axis = 1).reshape(-1, 1)
        if type(s) == type(None):
            return l/4
        else:
            new_s = s.copy()
            mask = isnan(s)
            new_s[mask] = l[mask]/4
        return new_s.reshape(-1, 1)


    @staticmethod
    def compute_v(chi, sigma, n):
        """
        Computes v, where
        v[i] = (1/sqrt(2*pi*sigma)) * e ^ (-(chi - (i+1))**2)/(2*sigma)

        Parameters
        ----------
            chi : int, float, ... | >= 1.
                The value of the random variable.
            sigma : int, float, ... | > 0.
                The value of the standard deviation.
            n : int | >= 1.
                The desired length of the vector v.

        Returns
        -------
            np.array, shape = (n, 1), which represents v.
        """
        return (1/sqrt(2*pi*sigma)) * (exp(-1 * ((chi - arange(1, n+1).reshape(-1, 1)) ** 2) / (2 * sigma)))


    @staticmethod
    def compute_prob_density(x, R, m, s):
        """
        Computes the probability densities for responses.

        Parameters
        ----------
            x : numpy.array, shape = (n, 1).
                A vector containing the new answer which is to be measured, where
                (chi_i) is the answer to the (i_th) question.
            R : numpy.array, dtype=float, shape = (n, max(q)), default : None.
                It is one part of the linear combination that forms the distro (1/gamma) * (R @ v).
            m : numpy.array, dtype=float, shape = (n, 1), default : None.
                The number of responses so far.
            s : numpy.array, dtype=float, default : None.
                The standard deviation of the distro at each modal. It controls
                the width of the distro at each model. In other words, if an
                answer is chosen, it affects how much the probability densities of
                nearby answers are affected.

        Returns
        -------
            numpy.array, shape = (n, 1), dtype="float64" that represents
            the probability density of x in a gaussian multimodal distribution
            given by
            p[i] = (1/(m * sqrt(2 * pi * s))) * (R[i, 0] * e ** (-((x-1) ** 2)/s) + R[i, 1] * e ** (-((x-2) ** 2)/s)
                                         + ... + R[i, n-1] * e ** (-((x-n) ** 2)/s))
        """
        # Initialize p
        p = zeros(x.shape, dtype="float64")

        # Loop through each question
        for i in range(R.shape[0]):
            # If the answers of the question are discrete
            if s[i, 0] == 0:
                p[i, 0] = R[i, int(x[i, 0]-1)]/m[i, 0]
            # Else, the answers of the question are continuous
            else:
                v = MultimodalDistroModel.compute_v(x[i], s[i], R.shape[1])
                p[i, 0] = (1/m[i, 0]) * (R[i, :].reshape(1, -1) @ v)


        # Return the prob density
        return p


    @staticmethod
    def compute_d(chi, sigma, n, displacement, direction):
        """
        Computes d- or d+, where
        d- = np.array([[erf((x[i] - displacement - 1) / (sqrt(s[i] * 2)))],
                       [erf((x[i] - displacement - 2) / (sqrt(s[i] * 2)))],
                       .
                       .
                       .,
                       [erf((x[i] - displacement - n) / (sqrt(s[i] * 2)))]])

        d+ = np.array([[erf((x[i] + displacement - 1) / (sqrt(s[i] * 2)))],
                       [erf((x[i] + displacement - 2) / (sqrt(s[i] * 2)))],
                       .
                       .
                       .,
                       [erf((x[i] + displacement - n) / (sqrt(s[i] * 2)))]])

        Parameters
        ----------
            chi : int, float, ... | >= 1.
                The value of the random variable.
            sigma : int, float, ... | > 0.
                The value of the standard deviation.
            n : int | >= 1.
                The desired length of the vector v.
            displacement: int, float, ... | > 0
            direction: str.
                If "-", d- is returned, and if "+", d+ is returned.

        Returns
        -------
            np.array, shape = (n, 1), which represents v.
        """
        if direction == "-":
            return erf((chi - displacement - arange(1, n+1).reshape(-1, 1)) / sqrt(2 * sigma))
        elif direction == "+":
            return erf((chi + displacement - arange(1, n+1).reshape(-1, 1)) / sqrt(2 * sigma))


    @staticmethod
    def compute_prob(x, R, m, s, neg_displacement, pos_displacement):
        """
        Computes the probability of responses being valid.

        Parameters
        ----------
            x : numpy.array, shape = (n, 1).
                A vector containing the new answer which is to be measured, where
                (chi_i) is the answer to the (i_th) question.
            R : numpy.array, dtype=float, shape = (n, max(q)), default : None.
                It is one part of the linear combination that forms the distro (1/gamma) * (R @ v).
            m : numpy.array, dtype=float, shape = (n, 1), default : None.
                The number of responses so far.
            s : numpy.array, dtype=float, default : None.
                The standard deviation of the distro at each modal. It controls
                the width of the distro at each model. In other words, if an
                answer is chosen, it affects how much the probability densities of
                nearby answers are affected.
            neg_displacement: np.array, dtype=float, shape=(n, 1).
                The displacement in the negative direction for integrating.
            pos_displacement: np.array, dtype=float, shape=(n, 1).
                The displacement in the positive direction for integrating.

        Returns
        -------
            numpy.array, shape = (n, 1), dtype="float64" that represents
            the probability of x being valid in a gaussian multimodal distribution
            given by
            p[i] = (1/(2*m)) * [R[i, 0] * erf((x[i] + displacement - 1) / (sqrt(s[i] * 2))) + R[i, 1] * erf(((x[i] + displacement - 2) / (sqrt(s[i] * 2))) + ... + R[i, n-1] * erf(((x[i] + displacement - n) / (sqrt(s[i] * 2)))]
                  - (1/(2*m)) * [R[i, 0] * erf((x[i] - displacement - 1) / (sqrt(s[i] * 2))) + R[i, 1] * erf(((x[i] - displacement - 2) / (sqrt(s[i] * 2))) + ... + R[i, n-1] * erf(((x[i] - displacement - n) / (sqrt(s[i] * 2)))]
            Vectorized form:
            p[i] = (1/(2*m)) * R[i, :] * (d2 - d1), where
                d2 = np.array([[erf((x[i] + displacement - 1) / (sqrt(s[i] * 2)))],
                               [erf((x[i] + displacement - 2) / (sqrt(s[i] * 2)))],
                               .
                               .
                               .,
                               [erf((x[i] + displacement - n) / (sqrt(s[i] * 2)))]])

                d1 = np.array([[erf((x[i] - displacement - 1) / (sqrt(s[i] * 2)))],
                               [erf((x[i] - displacement - 2) / (sqrt(s[i] * 2)))],
                               .
                               .
                               .,
                               [erf((x[i] - displacement - n) / (sqrt(s[i] * 2)))]])
        """
        # Initialize p
        p = zeros(x.shape, dtype="float64")

        # Loop through each question
        for i in range(R.shape[0]):
            # If the answers of the question are discrete
            if s[i, 0] == 0:
                p[i, 0] = R[i, int(x[i, 0]-1)]/m[i, 0]
            # Else, the answers of the question are continuous
            else:
                d1 = MultimodalDistroModel.compute_d(x[i, 0], s[i, 0], R.shape[1], displacement=neg_displacement[i, 0], direction="-")
                d2 = MultimodalDistroModel.compute_d(x[i, 0], s[i, 0], R.shape[1], displacement=pos_displacement[i, 0], direction="+")
                p[i, 0] = (1/(2*m[i, 0])) * (R[i, :].reshape(1, -1) @ (d2 - d1))

        # Return the prob
        return p


class DependencyMap:
    @staticmethod
    def handle_input(M, shape):
        if type(M) != type(None):
            _M = array(M)

            # ------- Self-dimension Errors:
            if _M.ndim == 1:
                _M = _M.reshape(1, -1).astype("float64")
            else:
                _M = _M.astype("float64")
        else:
            _M = None

        if type(shape) != type(None):
            _shape = tuple(shape)

        else:
            _shape = None

        return _M, _shape

    @staticmethod
    def handle_input_update(q_0, q_1):
        if type(q_0) != type(None):
            # Cast it to np.array
            _q_0 = array(q_0)

            # ------- Self-dimension Errors:
            if _q_0.ndim == 1:
                n = _q_0.size
            elif _q_0.ndim == 2:
                if (_q_0.shape[0] != 1) and (_q_0.shape[1] != 1):
                    raise ValueError(f"q_0 must be a vector of size (n, 1) or (1, n) or (n,), found {q_0.shape}")
                else:
                    n = (_q_0.shape[0]) if (_q_0.shape[0] != 1) else (_q_0.shape[1])
            else:
                raise ValueError(f"q_0 must be a vector of size (n, 1) or (1, n) or (n,), found {q_0.shape}")

            # Reshape q_0 to make sure it is a column vector and cast it to float64
            _q_0 = _q_0.reshape(n, 1).astype("float64")
        else:
            _q_0 = None

        if type(q_1) != type(None):
            # Cast it to np.array
            _q_1 = array(q_1)

            # ------- Self-dimension Errors:
            if _q_1.ndim == 1:
                n = _q_1.size
            elif _q_1.ndim == 2:
                if (_q_1.shape[0] != 1) and (_q_1.shape[1] != 1):
                    raise ValueError(f"q_1 must be a vector of size (n, 1) or (1, n) or (n,), found {q_1.shape}")
                else:
                    n = (_q_1.shape[0]) if (_q_1.shape[0] != 1) else (_q_1.shape[1])
            else:
                raise ValueError(f"q_1 must be a vector of size (n, 1) or (1, n) or (n,), found {q_1.shape}")

            # Reshape q_1 to make sure it is a column vector and cast it to float64
            _q_1 = _q_1.reshape(1, n).astype("float64")
        else:
            _q_1 = None

        return _q_0, _q_1

    def handle_input_M_to_R(q_1, R_width, m):
        if type(q_1) != type(None):
            # Cast it to np.array
            _q_1 = array(q_1)

            # ------- Self-dimension Errors:
            if _q_1.ndim == 1:
                n = _q_1.size
            elif _q_1.ndim == 2:
                if (_q_1.shape[0] != 1) and (_q_1.shape[1] != 1):
                    raise ValueError(f"q_1 must be a vector of size (n, 1) or (1, n) or (n,), found {q_1.shape}")
                else:
                    n = (_q_1.shape[0]) if (_q_1.shape[0] != 1) else (_q_1.shape[1])
            else:
                raise ValueError(f"q_1 must be a vector of size (n, 1) or (1, n) or (n,), found {q_1.shape}")

            # Reshape q_1 to make sure it is a column vector and cast it to float64
            _q_1 = _q_1.reshape(1, n).astype("float64")
        else:
            _q_1 = None

        if type(R_width) != type(None):
            if type(R_width) != type(1):
                raise ValueError(f"R_width must be an integer greater than zero, found {R_width}")

            if R_width <= 0:
                raise ValueError(f"R_width must be an integer greater than zero, found {R_width}")

            _R_width = R_width

        if type(m) != type(None):
            _m = array(m)

            # ------- Self-dimension Errors:
            if _m.ndim == 1:
                n = _m.size
            elif _m.ndim == 2:
                if min(_m.shape) != 1:
                    raise ValueError(f"m must be a vector of size (n, 1) or (1, n) or (n,), found {m.shape}")
                else:
                    n = max(_m.shape)
            else:
                raise ValueError(f"m must be a vector of size (n, 1) or (1, n) or (n,), found {m.shape}")

            # Reshape m to make sure it is a column vector and cast it to float64
            _m = _m.reshape(n, 1).astype("float64")
        else:
            _m = None

        return _q_1, _R_width, _m


class TimeResponseMeasure:
    pass


class ODResponseMeasure:
    @staticmethod
    def compute_max_distance(q, distance_measure):
        """
        Computes the max distance in the space defined by q, greedly, which means
        that it gets too close to the max distance but it is not guaranteed that
        we get to the absolute maximum.
        When tested, the maximum error was (-55e-3). It is notable that the
        acquired distance never exceeds the actual maximum distance.

        Parameters
        ----------
            q : numpy.array, shape = (n, 1).
                A vector containing the number of answers for each question, where
                (theta_i) is the number of answers for the (i_th) question.
            distance_measure : int.
                The distance measure used to meaesure distance between neighbours.
                It is one of:
                    (1): Euclidean.
                    (2): City Block.
                    (3): Cosine.

        Returns
        -------
            float: Represents the maximum possible distance
            (defined by the given distance measure) between two vectors
            in the space defined by q.
        """
        # Euclidean
        if distance_measure == 1:
            return float(norm(q-1, 2, axis=0))

        # City-block
        elif distance_measure == 2:
            return float(sum(q-1))

        # Cosine
        # Requires Modification
        elif distance_measure == 3:
            # Make sure there are at least two axes
            if not (q.size >= 2):
                return 0

            # Generate 2 corner points
            x, y = generate_2_corner_points(q)

            d = list()

            # Compute D = (x dot y), N = [[(norm(x, 2) ** 2), = (norm(y, 2) ** 2)]]
            D = x.T @ y
            N = array([[sum(x ** 2), sum(y ** 2)]])

            # Stack x and y together hotizontally -> [x y]
            XY = hstack((x, y))

            while(True):
                change_flag = 0
                # Iterate through XY horizontally (i.e. go through x then y)
                for j in range(XY.shape[1]):
                    # Iterate through XY vertically (i.e. go through each dimension)
                    for i in range(XY.shape[0]):
                        # Get (j_complement), which is the index of the other vector (i.e. if j=0 -> x then j_c=1 -> y)
                        j_c = int(not j)
                        # If the current element is minimum
                        if XY[i, j] == 1:
                            # Get D_max (x dot y if the current element is maximum)
                            # and N_max (the norm of (x if j = 0 and y if j = 1) if the current element is maximum)
                            D_max = D + (XY[i, j_c] * (q[i, :] - 1))
                            N_max = N[:, j] + ((q[i, :] ** 2) - 1)
                            # Get the difference value
                            diff = D_max * sqrt(N[:, j]) - D * sqrt(N_max)
                            # Change the current element to maximum if the difference value is negative
                            if float(diff) < 0:
                                XY[i, j] = q[i, :]
                                # The value of N[:, j] and D changes to N_max and D_max because the current element became max
                                D = D_max
                                N[:, j] = N_max
                                # Set the change flag to 1 so that another iteration (in the while loop) is executed
                                change_flag = 1
                        # Else (the current element is maximum)
                        else:
                            # Get D_min (x dot y if the current element is minimum)
                            # and N_max (the norm of (x if j = 0 and y if j = 1) if the current element is minimum)
                            D_min = D + (XY[i, j_c] * (1 - XY[i, j]))
                            N_min = N[:, j] + (1 - (XY[i, j] ** 2))
                            # Get the difference value
                            diff = D * sqrt(N_min) - D_min * sqrt(N[:, j])
                            # Change the current element to minimum if the difference value is positive
                            if float(diff) > 0:
                                XY[i, j] = 1
                                # The value of N[:, j] and D changes to N_min and D_min because the current element became min
                                D = D_min
                                N[:, j] = N_min
                                # Set the change flag to 1 so that another iteration (in the while loop) is executed
                                change_flag = 1
                # If no change happens, we got to our desired point
                if not change_flag:
                    break
            return float(ODResponseMeasure.compute_distances(XY[:, 0].reshape((-1, 1)), XY[:, 1].reshape((-1, 1)), 3, False))

        # Otherewise
        raise ValueError("distance_measure is an integer that has to be one of: (1) Euclidean. (2) City Block. (3) Cosine.")

    @staticmethod
    def compute_distances(x_new, X_train, distance_measure, zero_based):
        """
        Computes the distances between the new response and the previous answers.

        Parameters
        ----------
            x_new : numpy.array, shape = (n, 1).
                A vector containing the new answer which is to be measured, where
                (chi_i) is the answer for the (i_th) qustion.
            X_train : numpy.array, shape = (n, m).
                A matrix containing the previous answers, where (chi_i,j) is the
                answer for the (i_th) question of the (j_th) respondent.
            distance_measure : int.
                The distance measure used to meaesure distance between neighbours.
                It is one of:
                    (1): Euclidean.
                    (2): City Block.
                    (3): Cosine.
            zero_based : boolean.
                Set it True if the indices of your answers begin with 0, and
                set it False if they begin with 1.
                (i.e. if your first answer is indexed 0, set it True)
        Returns
        -------
            numpy.array, shape = (1, m), where the (j_th) entry represents the
            distance between the new response and the (j_th) previous response.
        """
        # Euclidean
        if distance_measure == 1:
            delta = x_new - X_train
            return norm(delta, 2, axis=0).reshape(1, -1)

        # City-block
        elif distance_measure == 2:
            delta = x_new - X_train
            return sum(abs(delta), axis=0).reshape(1, -1)

        # Cosine
        elif distance_measure == 3:
            # Prepare for division by casting x_new and X_train to float64
            x_new = array(x_new, dtype='float64')
            X_train = array(X_train, dtype='float64')
            # Handle the origin point
            if zero_based:
                x_new = x_new + 1
                X_train = X_train + 1
            # Compute the numerator and denominator
            num = (x_new.T @ X_train).reshape(1, -1)
            denom = (norm(x_new, 2, axis=0) * norm(X_train, 2, axis=0)).reshape(1, -1)
            # Initialize the array that holds the results
            distances = ones((1, X_train.shape[1]), dtype='float64')
            # Compute the cosine distances
            divide(num, denom, out=distances, where=(denom != 0))
            distances = 1 - distances
            return distances

        # Otherewise
        raise ValueError("distance_measure is an integer that has to be one of: (1) Euclidean. (2) City Block. (3) Cosine.")

    @staticmethod
    def cumulate_distances(distances, k):
        """
        Cumulates the distances between the new response and the k nearest responses.

        Parameters
        ----------
            distances : numpy.array, shape = (1, m)
                The (j_th) entry represents the distance between
                the new response and the (j_th) previous response.
            k : int, greater than zero, and shouldn't exceed distances.size
                The number of neighbours.

        Returns
        -------
            float representing the cumulative distance between the new response
            and the k nearest responses.
        """
        return float(sum(sort(distances)[:, :k]))
