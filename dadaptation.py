import numpy as np
import matplotlib.pyplot as plt

a = 1
b= 5

def func(x: np.ndarray) -> float:
    """
    calculate the nemirovski function

    :param x: the input vector
    :return:
    """
    return a * np.max(np.abs(x - b))


def calc_der(x:np.ndarray):
    """
    This method calculates the derivative of the Nemirovski function
    :param x:
    :return:
    """
    idx = np.argmax(np.abs(x-b))
    vect = np.zeros(len(x))
    vect[idx] = 1
    return a * np.sign(x[idx]-b) * vect


def calculate_vect(lambda_lst, vect_lst):
    dem = 1 / np.sum(lambda_lst)
    vect = 0
    for i in range(len(lambda_lst)):
        vect += lambda_lst[i] * vect_lst [i]
    return vect * dem

def gradient_decent_with_d_adaptation(start, n_iter, initial_d=1e-6):
    vector = start
    total_der = 0
    err = [func(vector)]
    s=0
    d = initial_d
    g_mult_labmda = 0
    lambda_sum = 0
    lambda_vect_sum = 0
    d_lst = []
    for _ in range(n_iter):
        der = calc_der(vector)  # calculate g
        total_der += np.linalg.norm(der) ** 2  # sum of derivitie norms squared
        lambda_k = d / np.sqrt((a ** 2 + total_der))
        lambda_sum += lambda_k
        g_mult_labmda += (lambda_k ** 2) * (np.linalg.norm(der) ** 2)
        s = s + lambda_k * der
        new_d = ((np.linalg.norm(s) ** 2) - g_mult_labmda) / (2 * np.linalg.norm(s))
        d = max(new_d, d)
        d_lst.append(d)
        vector = vector - lambda_k * der
        lambda_vect_sum += vector * lambda_k
        err.append(func((np.copy(1/lambda_sum) * lambda_vect_sum)))
    return vector, err,d_lst

if __name__ == '__main__':
    d_lst = [1e-8, 1e-6, 1e-2, 1, 5, 50]
    iterations = 60000
    plt.figure()
    all_errors = {}
    all_ds = {}
    vector_size = 100
    for j in range(len(d_lst)):
        vect, err, d_vect = gradient_decent_with_d_adaptation(start=np.zeros(vector_size), n_iter=iterations, initial_d=d_lst[j])
        all_errors[d_lst[j]] = err[-1]
        all_ds[d_lst[j]] = d_vect[-1]
        plt.plot(np.arange(len(err)), err, label=f"d_0 = {d_lst[j]}")
    dist_to_optimal_solution = np.linalg.norm(np.ones(vector_size) * b)
    # plt.plot(np.arange(len(d_vect)), np.ones(len(d_vect)) * dist_to_optimal_solution, label="real D value")
    plt.xlabel("iteration number")
    plt.ylabel("sqaured error")
    plt.title("squared error as a function of iteration number")
    plt.legend()
    plt.show()
