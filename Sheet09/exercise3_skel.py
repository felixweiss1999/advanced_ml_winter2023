import numpy as np
import matplotlib.pyplot as plt
from time import sleep

n = 10
n_range = np.arange(1, n+1)
T = np.diag(2*n_range + 1) - \
    np.diag(n_range[1:], -1) - \
    np.diag(n_range[1:], 1)
T[n-1, n-1] = n
b = np.eye(n, 1).reshape(n)

def compute_eta(p, g):

    # TODO Implement the computation of the step length
    # that minimizes the 2-norm of the next gradient
    return eta

def newton(x0: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the minimum of the function
    \[
        f: \\mathbb{R}^n \\rightarrow \\mathbb{R}, \\quad
        f(\\mathbf{x}) = \\frac{1}{2} \\left((x_1 - 1)^2 +
        \sum_{k = 2}^n k(x_k - x_{k-1})^2\\right)
    \]
    with Newton's method.

    Parameters
    ----------
    x0 : np.ndarray
        Start guess.
    m : int
        Maximal number of steps of Newton's method.

    Returns
    -------
    np.ndarray
        Approximation to the point of minimization
        computed by Newton's method.
    np.ndarray
        Array containing the error of each step of
        Newton's method.
    """
    x = x0
    err = np.zeros(m+1, dtype=float)
    err[0] = np.linalg.norm(x-1)
    for k in range(m):

        # TODO Implement the loop of Newton's method.
        # See, e.g., p. 57 of the lecture notes for more information.
        pass

    return x, err

def bfgs(x0: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the minimum of the function
    \[
        f: \\mathbb{R}^n \\rightarrow \\mathbb{R}, \\quad
        f(\\mathbf{x}) = \\frac{1}{2} \\left((x_1 - 1)^2 +
        \sum_{k = 2}^n k(x_k - x_{k-1})^2\\right)
    \]
    with BFGS.

    Parameters
    ----------
    x0 : np.ndarray
        Start guess.
    m : int
        Maximal number of steps of the BFGS method.

    Returns
    -------
    np.ndarray
        Approximation to the point of minimization
        computed by the BFGS method.
    np.ndarray
        Array containing the error of each step of
        the BFGS method.
    np.ndarray
        Last approximated inverse.
    """
    n = x0.size
    x = x0
    err = np.zeros(m+1, dtype=float)
    err[0] = np.linalg.norm(x-1)
    I = np.eye(n)
    H = I
    for k in range(m):
        # TODO Implement the loop of the BFGS method.
        # See, e.g., p. 144 of the lecture notes, or
        # https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm
        pass
    return x, err, H

def get_lbfgs_update(k: int, Y: np.ndarray, S: np.ndarray, rho: np.ndarray,
                     g: np.ndarray, m: int, modified: bool = True) -> np.ndarray:
    """
    Compute the update for the L-BFGS method. This function implicitly
    applies the BFGS inverse of the Hessian of the function
    \[
        f: \\mathbb{R}^n \\rightarrow \\mathbb{R}, \\quad
        f(\\mathbf{x}) = \\frac{1}{2} \\left((x_1 - 1)^2 +
        \sum_{k = 2}^n k(x_k - x_{k-1})^2\\right)
    \]
    to the gradient in a step.

    Parameters
    ----------
    k : int
        Current step of the L-BFGS method.
    Y : np.ndarray
        Previous vectors y_j assembled columnwise in
        a matrix.
    S : np.ndarray
        Previous vectors s_j assembled columnwise in
        a matrix.
    rho : float
        Previous values $\\rho_j$ assembled in a vector.
    g : np.ndarray
        Gradient of the function f.
    m : int
        Number of approximation steps for the inverse.
    modified : bool, optional
        _description_, by default True

    Returns
    -------
    np.ndarray
        Update of the L-BFGS method, which is an approximation
        of the inverse Hession applied to the gradient.
    """
    p = -g
    alpha = np.zeros(k)
    for j in range(k-1, max(-1, k-m), -1):
        # TODO Apply the backwards iteration
        # of the Algorithm from the exercise
        # sheet.
        pass

    if (k > 0) & modified:
        # TODO multiply p with gamma != 1
        # if the modified flag is set to True.
        pass

    for j in range(max(0, k-m), k):
        # TODO Apply the forward iteration of
        # the algorithm from the exercise sheet.
        pass

    return p

def lbfgs(x0: np.ndarray, m: int, steps: int, modified: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the minimum of the function
    \[
        f: \\mathbb{R}^n \\rightarrow \\mathbb{R}, \\quad
        f(\\mathbf{x}) = \\frac{1}{2} \\left((x_1 - 1)^2 +
        \sum_{k = 2}^n k(x_k - x_{k-1})^2\\right)
    \]
    with the L-BFGS method.

    Parameters
    ----------
    x0 : np.ndarray
        Start guess.
    m : int
        Maximal number of steps of the BFGS method.
    steps : int
        Number of steps for the update approximation.
    modified : bool, optional
        Flag determining the scaling parameter
        $\\gamma$ in the computation of the update,
        by default False.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        _description_
    """

    x = x0
    err = np.zeros(m+1, dtype=float)
    err[0] = np.linalg.norm(x-1)
    rhovec = np.zeros(m)
    Y = np.zeros((n, m))
    S = np.zeros((n, m))
    for k in range(m):
        # TODO Implement the loop for the L-BFGS method.
        pass

    return x, err

if __name__ == '__main__':

    print("matrix T =\n", T)
    print("right-hand side b =\n", b)
    sleep(2)

    # parameters for the tests
    x0 = np.random.randn(n)
    m = 100
    I = np.eye(n)

    x, Newton = newton(x0, m)
    print("solution by Newton =\n", x)
    sleep(2)

    x, BFGS, H = bfgs(x0, m)
    print("solution by BFGS =\n", x)

    norm_inverse = np.linalg.norm(I - H @ T)
    print("Error in inverse =\n", norm_inverse)
    sleep(2)

    stepmin = 1
    stepmax = 30
    LBFGS = np.zeros((m+1, stepmax-stepmin+1))
    for steps in range(stepmin, stepmax+1):
        x, LBFGS[:, steps-stepmin
                 ] = lbfgs(x0, m, steps, modified=False)

    print("solution by L-BFGS (plain) =\n", x)
    sleep(2)

    plt.semilogy(np.arange(m+1), Newton, 'k-',
                 np.arange(m+1), BFGS, 'b-',
                 np.arange(m+1), LBFGS, 'c-.')
    plt.title('Newton vs. BFGS / L-BFGS')
    plt.legend(['Newton', 'BFGS', 'L-BFGS (plain)'], loc='upper right')
    plt.savefig("LBFGSplain.pdf", bbox_inches='tight')
    plt.show()

    stepmin = 1
    stepmax = 30
    LBFGS = np.zeros((m+1, stepmax-stepmin+1))
    for steps in range(stepmin, stepmax+1):
        x, LBFGS[:, steps-stepmin
                 ] = lbfgs(x0, m, steps, modified=True)

    print("solution by L-BFGS (modified) =\n", x)
    sleep(2)

    plt.semilogy(np.arange(m+1), Newton, 'k-',
                 np.arange(m+1), BFGS, 'b-',
                 np.arange(m+1), LBFGS, 'c-.')
    plt.title('Newton vs. BFGS / L-BFGS')
    plt.legend(['Newton', 'BFGS', 'L-BFGS (modified)'], loc='upper right')
    plt.savefig("LBFGSmodified.pdf", bbox_inches='tight')
    plt.show()
