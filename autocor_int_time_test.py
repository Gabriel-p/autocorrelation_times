
import numpy as np
import emcee
from statsmodels.tsa.ar_model import AR
import matplotlib.pyplot as plt


def main():

    # Define number of datasets to process (M). The level of correlation of
    # each set is given by the STDDEV value in the list below
    # (between .1 and 5.)
    M = 20
    std = np.linspace(.1, 5., M)

    ess_all = [[] for _ in range(6)]
    for _ in range(M):

        # Generate synthetic data.
        data, lags, N = dataLags(std[_])

        # Obtain integrated autocorrelation time with all methods.
        iat0 = iat_emcee(data)
        ess_all[0].append(iat0)
        print("emcee: {:.2f}".format(iat0))
        iat1 = iat_ISE(data, lags)
        ess_all[1].append(iat1)
        print("ISE  : {:.2f}".format(iat1))
        iat2 = iat_BM(data)
        ess_all[2].append(iat2)
        print("BM   : {:.2f}".format(iat2))
        iat3, dummy = ACOR(data)
        ess_all[3].append(iat3)
        print("ACOR : {:.2f}".format(iat3))
        iat4 = iat_AR(data, lags)
        ess_all[4].append(iat4)
        print("AR(p): {:.2f}".format(iat4))
        # For this method, divide by the sample size to obtain the IAT.
        mESS = multiESS(data.reshape(N, 1))
        ess_all[5].append(float(N) / mESS)
        print("mESS: {:.2f}".format(float(N) / mESS))

        print("")

        # Plot the data
        # plt.scatter(range(N), data, s=3, label="Std={:.2f}".format(std[_]))
        # plt.legend()
        # plt.show()

    # Plot the final results
    makePlot(M, std, ess_all)


def makePlot(M, std, ess_all):
    plt.style.use('seaborn')
    plt.title("Methods for IAT (decreasing correlation to the right)")
    plt.ylabel("IAT")
    plt.xlabel("stddev")
    plt.plot(range(M), ess_all[0], label="emcee")
    plt.plot(range(M), ess_all[1], label="ISE", ls='--')
    plt.plot(range(M), ess_all[2], label="BM", ls='-.')
    plt.plot(range(M), ess_all[3], label="ACOR", ls=':')
    plt.plot(range(M), ess_all[4], label="AR(p)")
    plt.plot(range(M), ess_all[5], label="mESS", ls='--')
    plt.axhline(y=1., c='r', lw=1.5)
    plt.legend()
    plt.xticks(range(M), np.round(std, 2))
    plt.show()


def sampleLags(x):
    """
    Generate sample lags.
    """
    x = x - x.mean()
    lags = np.correlate(x, x, mode='full')[-x.size:]
    # Normalization
    lags /= lags.max()
    return lags


def dataLags(std):
    # Some random correlated+noise data
    N = 10000
    noise = np.random.normal(0, std, N)
    data = np.sin(np.deg2rad(range(N))) + noise

    # Lags.
    lags = sampleLags(data)

    return data, lags, N


def iat_emcee(data):
    """
    emcee integrated autocorrelation time.
    """
    try:
        iat = emcee.autocorr.integrated_time(data)
    except Exception:
        # Chain was too short
        print("emcee error: chain is too short")
        iat = np.nan
    return iat


def iat_ISE(data, lags):
    """
    Initial sequence estimators integrated autocorrelation time,
    Thompson (2010) Eq (4) with number of lags selected according to the rule
    given in Sokal 1996 (see https://dfm.io/posts/autocorr/, after Eq 10)
    """
    N = len(data)
    iat = -1.
    for i in [.001, .002, .005, .01, .02, .05, .075, .1, .2, .25, .3, .35, .5]:
        M = int(i * N)
        iat = 1. + 2. * np.sum(lags[:M])
        # Stopping condition
        if M > iat * 5. and iat > 0.:
            # print("   {}".format(i * 100.))
            break
    return iat


def iat_BM(data):
    """
    Batch means integrated autocorrelation time,
    Thompson (2010) Eq (2)
    """
    b_s = int(len(data) ** (1. / 3.))
    data_split = np.array_split(data, b_s)
    b_means = []
    for batch in data_split:
        b_means.append(np.mean(batch))
    var_b = np.var(b_means)
    var_d = np.var(data)
    iat = len(data_split[0]) * var_b / var_d
    return iat


def iat_AR(data, lags):
    """
    AR(p) process integrated autocorrelation time,
    Thompson (2010) Eq (7)

    https://machinelearningmastery.com/
        autoregression-models-time-series-forecasting-python/
    """
    model = AR(data)
    model_fit = model.fit(ix='aic', trend='nc')
    # Number of lags 'k_ar' chosen by the AIC method.
    n_p = model_fit.k_ar
    # Fitted coefficients
    coeffs = model_fit.params

    iat = (1. - np.dot(lags[:n_p], coeffs)) / (1. - np.sum(coeffs)) ** 2
    return iat


def ACOR(X, TAUMAX=2, WINMULT=5, MINFAC=5):
    """
    Source: https://www.math.nyu.edu/faculty/goodman/software/acor/

    Compute tau directly only if tau < TAUMAX. Otherwise compute tau using the
    pairwise sum series
    TAUMAX = 2
    Compute autocovariances up to lag s = WINMULT*TAU
    WINMULT = 5
    The autocovariance array is double C[MAXLAG+1] so that C[s] makes sense
    for s = MAXLAG.
    MAXLAG = TAUMAX * WINMULT
    # Stop and print an error message if the array is shorter
    # than MINFAC * MAXLAG.
    MINFAC = 5

    """

    MAXLAG = TAUMAX * WINMULT

    # Subtract the mean
    X = X - np.mean(X)
    L = len(X)

    if L < MINFAC * MAXLAG:
        print(
            "Acor error: The autocorrelation time is too long relative "
            "to the variance.")

    C = np.zeros(MAXLAG + 1)
    # Compute the autocovariance function
    iMax = L - MAXLAG
    for i in range(iMax):
        for s in range(MAXLAG + 1):
            # first the inner products
            C[s] += X[i] * X[i + s]

    # then the normalization
    C /= iMax
    # for s in range(MAXLAG + 1):
    #     C[s] /= iMax

    # The "diffusion coefficient" is the sum of the autocovariances
    # The rest of the C[s] are double counted since C[-s] = C[s].
    D = C[0] + 2. * np.sum(C[1:])

    # The standard error bar formula, if D were the complete sum.
    sigma = np.sqrt(D / L)
    # A provisional estimate, since D is only part of the complete sum.
    tau = D / C[0]

    # Stop if the D sum includes the given multiple of tau. This is the self
    # consistent window approach.
    if tau * WINMULT < MAXLAG:
        return tau, sigma
    else:
        # If the provisional tau is so large that we don't think tau is
        # accurate, apply the acor procedure to the pairwase sums of X.

        # The pairwise sequence is half the length (if L is even)
        Lh = L / 2

        j1, j2 = 0, 1
        for i in range(Lh):
            X[i] = X[j1] + X[j2]
            j1 += 2
            j2 += 2

        tau, sigma = ACOR(X[:Lh])

        # Reconstruct the fine time series numbers from the coarse series
        # numbers.
        D = .25 * sigma * sigma * L
        # As before, but with a corrected D.
        tau = D / C[0]
        # As before, again.
        sigma = np.sqrt(D / L)

        return tau, sigma


def multiESS(X, b='sqroot', Noffsets=10, Nb=None):
    """
    Compute multivariate effective sample size of a single Markov chain X,
    using the multivariate dependence structure of the process.

    X: MCMC samples of shape (n, p)
    n: number of samples
    p: number of parameters

    b: specifies the batch size for estimation of the covariance matrix in
       Markov chain CLT. It can take a numeric value between 1 and n/2, or a
       char value between:

    'sqroot'    b=floor(n^(1/2)) (for chains with slow mixing time; default)
    'cuberoot'  b=floor(n^(1/3)) (for chains with fast mixing time)
    'lESS'      pick the b that produces the lowest effective sample size
                for a number of b ranging from n^(1/4) to n/max(20,p); this
                is a conservative choice

    If n is not divisible by b Sigma is recomputed for up to Noffsets subsets
    of the data with different offsets, and the output mESS is the average over
    the effective sample sizes obtained for different offsets.

    Nb specifies the number of values of b to test when b='less'
    (default NB=200). This option is unused for other choices of b.

    Original source: https://github.com/lacerbi/multiESS

    Reference:
    Vats, D., Flegal, J. M., & Jones, G. L. "Multivariate Output Analysis
    for Markov chain Monte Carlo", arXiv preprint arXiv:1512.07713 (2015).

    """

    # MCMC samples and parameters
    n, p = X.shape

    if p > n:
        raise ValueError(
            "More dimensions than data points, cannot compute effective "
            "sample size.")

    # Input check for batch size B
    if isinstance(b, str):
        if b not in ['sqroot', 'cuberoot', 'less']:
            raise ValueError(
                "Unknown string for batch size. Allowed arguments are "
                "'sqroot', 'cuberoot' and 'lESS'.")
        if b != 'less' and Nb is not None:
            raise Warning(
                "Nonempty parameter NB will be ignored (NB is used "
                "only with 'lESS' batch size B).")
    else:
        if not 1. < b < (n / 2):
            raise ValueError(
                "The batch size B needs to be between 1 and N/2.")

    # Compute multiESS for the chain
    mESS = multiESS_chain(X, n, p, b, Noffsets, Nb)

    return mESS


def multiESS_chain(Xi, n, p, b, Noffsets, Nb):
    """
    Compute multiESS for a MCMC chain.
    """

    if b == 'sqroot':
        b = [int(np.floor(n ** (1. / 2)))]
    elif b == 'cuberoot':
        b = [int(np.floor(n ** (1. / 3)))]
    elif b == 'less':
        b_min = np.floor(n ** (1. / 4))
        b_max = max(np.floor(n / max(p, 20)), np.floor(np.sqrt(n)))
        if Nb is None:
            Nb = 200
        # Try NB log-spaced values of B from B_MIN to B_MAX
        b = set(map(int, np.round(np.exp(
            np.linspace(np.log(b_min), np.log(b_max), Nb)))))

    # Sample mean
    theta = np.mean(Xi, axis=0)
    # Determinant of sample covariance matrix
    if p == 1:
        detLambda = np.cov(Xi.T)
    else:
        detLambda = np.linalg.det(np.cov(Xi.T))

    # Compute mESS
    mESS_i = []
    for bi in b:
        mESS_i.append(multiESS_batch(Xi, n, p, theta, detLambda, bi, Noffsets))
    # Return lowest mESS
    mESS = np.min(mESS_i)

    return mESS


def multiESS_batch(Xi, n, p, theta, detLambda, b, Noffsets):
    """
    Compute multiESS for a given batch size B.
    """

    # Compute batch estimator for SIGMA
    a = int(np.floor(n / b))
    Sigma = np.zeros((p, p))
    offsets = np.sort(list(set(map(int, np.round(
        np.linspace(0, n - np.dot(a, b), Noffsets))))))

    for j in offsets:
        # Swapped a, b in reshape compared to the original code.
        Y = Xi[j + np.arange(a * b), :].reshape((a, b, p))
        Ybar = np.squeeze(np.mean(Y, axis=1))
        Z = Ybar - theta
        for i in range(a):
            if p == 1:
                Sigma += Z[i] ** 2
            else:
                Sigma += Z[i][np.newaxis, :].T * Z[i]

    Sigma = (Sigma * b) / (a - 1) / len(offsets)
    mESS = n * (detLambda / np.linalg.det(Sigma)) ** (1. / p)

    return mESS


if __name__ == '__main__':
    main()
