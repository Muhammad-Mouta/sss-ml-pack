import numpy as np

def generate_2_corner_points(q):
    """
    Generates two asymetric corner points in the space defined by q.

    Parameters
    ----------
        q : numpy.array, shape = (n, 1).
            A vector containing the number of answers for each question, where
            (theta_i) is the number of answers for the (i_th) question.

    Returns
    -------
        a tuple of numpy.array, shape = (n, 1): represents the coordinates of
        the two corners. The generated corners have the following specifications:
        - If the lengths of dimensions in q are all equal, let this length
          be (L), then x = (L L L .... L 1).T and y = (1 1 1 ... 1 L).T
        - Else, let the maximum length in q be L at locations C = {gamma_1, gamma_2, gamma_3, ... }
          then x = (chi_1 chi_2 chi_3 ... chi_n), where chi_i = L if i in C, 1 otherwise
          and y = (psi_1 psi_2 psi_3 ... psi_n), where psi_i = q[i] if i is not in C, 1 otherwise
    """
    if q.shape[0] == 1:
        raise ValueError("Q has to be of size (n, 1), where n > 1 to generate 2 corner points")

    # Get the dimension with the maximum length
    max_dim = np.amax(q)

    # Initialize x and y
    x = np.ones(q.shape)
    y = np.ones(q.shape)

    # Compute the two corner points
    if all(q == max_dim):
        x[:int(q.shape[0] / 2), :] = max_dim
        y[int(q.shape[0] / 2):, :] = max_dim
    else:
        x[(q == max_dim)] = max_dim
        #y[(q != max_dim)] = q[(q != max_dim)]

    return x, y


def compute_cosine_max_distance_brute_force(q):
    # from ml_pack.test.response_measures_utils_test import compute_cosine_max_distance_brute_force as ccm
    # Get all possible corners
    # Initialize a q.shape[0] * (2 ** q.shape[0]) array
    c = int(2 ** q.shape[0])
    pb = np.zeros((q.shape[0], c))
    for i in range(pb.shape[0]):
        c = int(c / 2)
        v = 0
        for j in range(0, pb.shape[1], c):
            pb[i, j:j+c] = v
            v = 1 if (v==0) else 0
        pb[i, :] = pb[i, :] * q[i, :]
        pb[pb == 0] = 1

    # Get the distance between each point and all the points
    d = np.zeros((pb.shape[1], pb.shape[1]))

    def ccd(x_new, X_train):
        # Compute the numerator and denominator
        num = (x_new.T @ X_train).reshape(1, -1)
        denom = (np.linalg.norm(x_new, 2, axis=0) * np.linalg.norm(X_train, 2, axis=0)).reshape(1, -1)
        # Initialize the array that holds the results
        distances = np.ones((1, X_train.shape[1]), dtype='float64')
        # Compute the cosine distances
        np.divide(num, denom, out=distances, where=(denom != 0))
        distances = 1 - distances
        return distances

    for i in range(pb.shape[1]):
        d[i+1:, i] = ccd(pb[:, i], pb[:, i+1:])

    print(np.unravel_index(np.argmax(d), d.shape))
    return np.amax(d)


def generate_random_sample(sample_size, q_count=58, q_size=5, method="normal", return_parameters=False):
    if method == "normal":
        # Initialize mean and std deviation
        mu = np.random.choice([1, 2, 3, 4, 5], size=(q_count, 1), p=[0.1, 0.2, 0.4, 0.2, 0.1])
        s = np.random.randint(1, q_size+1, size=(q_count, 1))
        parameters = (mu, s)

        # Generate random sample
        sample = np.round(np.random.normal(np.ones((q_count, sample_size)) * mu, np.ones((q_count, sample_size)) * s))

    elif method == "exponential":
        # Initialize beta
        beta = np.random.randint(1, q_size+1, size=(q_count, 1))
        parameters = beta

        # Generate random sample
        sample = np.round(np.random.exponential(beta, size=((q_count, sample_size))))

    elif method == "chisquare":
        # Initialize the degree of freedom
        df = np.random.randint(1, q_size+1, size=(q_count, 1))
        parameters = df

        # Generate random sample
        sample = np.round(np.random.chisquare(df, size=((q_count, sample_size))))


    sample[sample < 1] = 1
    sample[sample > q_size] = q_size

    return (sample, parameters) if return_parameters else sample


def count_responses(responses, q_size=5):
    # Initialize the count array
    count = np.zeros((responses.shape[0], q_size))

    # Populate it one by one
    for i in range(responses.shape[0]):
        for j in range(q_size):
            count[i, j] = np.count_nonzero(responses[i, :] == j+1)

    return count
