# Proxy for the Mtm of swaps :

project_path = r'C:\Users\omirinioui\PycharmProjects\Projet Quadrature bayésienne'

lib_path = project_path + '\libraries'
port_path = project_path + '\Data\portfolio'
zc_curve_path = project_path + '\Data\ZC Curve'
diff_path = project_path + '\Data\Diffusion'
MC_path = project_path + '\Results\Full MC'
img_path = project_path + '\Results\Images'

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import numpy as np
import pandas as pd

import numpy as np

from time import time
from datetime import timedelta

# importing pacakge
import sys

sys.path.append(lib_path)

from main import *
from FinancialData import *
from Chebychev import *


def GPR_Mtm_ploter(t, diff, irs, ZC, nodes_nbr, diag=False):
    train_range = (min(diff.X(t)), max(diff.X(t)))

    # Pricing to train
    train_points = np.linspace(train_range[0], train_range[1], nodes_nbr)
    mtm_points = irs.Mtm_mono(t, train_points, ZC, diff.get_LGM_params())

    # Train
    gpr_model = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
    gpr_model.fit(train_points.reshape(nodes_nbr, 1), mtm_points.reshape(nodes_nbr, 1))

    # Mtm Proxy
    Mtm = gpr_predictor(gpr_model, np.sort(diff.X(t)).reshape(diff.nbr_scenarios, 1))
    Mtm_true = irs.Mtm_mono(t, np.sort(diff.X(t)), ZC, diff.get_LGM_params())

    if diag:
        plt.plot(Mtm_true, Mtm_true, c='red', label='true')
        plt.scatter(Mtm_true, Mtm.reshape(diff.nbr_scenarios), label='GPR')

        plt.xlabel('True Mtm', fontsize=15)
        plt.ylabel('GPR', fontsize=15)
        plt.legend()
        plt.title(str(nodes_nbr) + ' training points', fontdict={'fontsize': 16})
        plt.show()
    else:
        N = 400
        x_range = np.linspace(train_range[0], train_range[1], N)
        plt.plot(x_range, irs.Mtm_mono(t, x_range, ZC, diff.get_LGM_params()), label='True')

        plt.scatter(train_points, mtm_points, c='r')

        plt.plot(np.sort(diff.X(t)), Mtm.reshape(diff.nbr_scenarios), label='GPR')

        plt.legend()
        plt.xlabel('$X_{t}$')
        plt.ylabel(' Mark-to-Market')
        plt.title(str(nodes_nbr) + ' training points')
        plt.show()


def Chebyshev_Mtm_ploter(t, diff, irs, ZC, nodes_nbr, diag=False):
    train_range = (min(diff.X(t)), max(diff.X(t)))

    # Pricing to train
    train_points = np.sort(Chebyshev_points(train_range[0], train_range[1], nodes_nbr))
    mtm_points = irs.Mtm_mono(t, train_points, ZC, diff.get_LGM_params())

    # Mtm Proxy
    Mtm = np.array([eval_Barycentric(mtm_points, train_points, x) for x in np.sort(diff.X(t))])
    Mtm_true = irs.Mtm_mono(t, np.sort(diff.X(t)), ZC, diff.get_LGM_params())

    if diag:
        plt.plot(Mtm_true, Mtm_true, c='red', label='true')
        plt.scatter(Mtm_true, Mtm, label='Chebyshev')

        plt.xlabel('True Mtm', fontsize=15)
        plt.ylabel('Chebyshev', fontsize=15)
        plt.legend()
        plt.title(str(nodes_nbr) + ' training points', fontdict={'fontsize': 16})
        plt.show()

    else:
        N = 400
        x_range = np.linspace(train_range[0], train_range[1], N)
        plt.plot(x_range, irs.Mtm_mono(t, x_range, ZC, diff.get_LGM_params()), label='true')

        plt.scatter(train_points, mtm_points, c='r')
        plt.plot(np.sort(diff.X(t)), Mtm, label='Chebyshev')

        plt.legend()
        plt.xlabel('$X_{t}$')
        plt.ylabel(' Mark-to-Market')
        plt.title(str(nodes_nbr) + ' training points')
        plt.show()


def Chebyshev_Mtm(t, diff, irs, ZC, nodes_nbr):
    train_range = (min(diff.X(t)), max(diff.X(t)))

    # Pricing to train
    train_points = np.sort(Chebyshev_points(train_range[0], train_range[1], nodes_nbr))
    mtm_points = irs.Mtm_mono(t, train_points, ZC, diff.get_LGM_params())

    # Mtm Proxy
    Mtm = np.array([eval_Barycentric(mtm_points, train_points, x) for x in np.sort(diff.X(t))])
    Mtm_true = irs.Mtm_mono(t, np.sort(diff.X(t)), ZC, diff.get_LGM_params())

    return Mtm


def GPR_Mtm(t, diff, irs, ZC, nodes_nbr, diag=False):
    train_range = (min(diff.X(t)), max(diff.X(t)))

    # Pricing to train
    train_points = np.linspace(train_range[0], train_range[1], nodes_nbr)
    mtm_points = irs.Mtm_mono(t, train_points, ZC, diff.get_LGM_params())

    if (mtm_points == np.zeros(nodes_nbr)).all():
        return 0

    # Train
    gpr_model = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
    gpr_model.fit(train_points.reshape(nodes_nbr, 1), mtm_points.reshape(nodes_nbr, 1))

    # Mtm Proxy
    Mtm = gpr_predictor(gpr_model, np.sort(diff.X(t)).reshape(diff.nbr_scenarios, 1))

    return Mtm.reshape(diff.nbr_scenarios)


ZCR = ZC_Data_extractor(zc_curve_path)
ZC = zero_coupon_bonds(ZCR, 3)  # We chose 3 as degree of the Spline Interpolation

# load portfolio data
with open(port_path + '\port400irs1fx', 'rb') as f1:
    portfolio = pickle.load(f1)

# Set Maturity
T = portfolio.last_maturity

# load diffusion data
with open(diff_path + '\MC60000p400swap1fx', 'rb') as f1:
    diff = pickle.load(f1)

# LGM params
sig = 0.02
lam = 0.015

# Diffusion params
T = portfolio.last_maturity
n = int(T * 360) + 1
NBR_SCENARIOS = 60000

tt = time()
diff = Diffusion(0, T, n, sig, lam, NBR_SCENARIOS, pb_measure='Terminal at t')
print('Diffusion Time', "{}".format(str(timedelta(seconds=round(time() - tt)))))

def quad_norm(x):
    return np.sum(np.sqrt(x)) ** 2


def beta(t, T, lam):
    return (1 - np.exp(-lam * (T - t))) / lam


def A(t, T, lam, sig, ZC):
    zcb_t, zcb_T = ZC.initial_zcb_curve(t), ZC.initial_zcb_curve(T)
    beta_ = beta(t, T, lam)
    phi = (sig ** 2) * (1 - np.exp(-2 * lam * t)) / (2 * lam)

    return (zcb_T / zcb_t) * np.exp(-0.5 * beta_ ** 2 * phi)


def zcb_law_lgm(t, T, lam, sig, ZC):
    mu_X_t = 0
    std_X_t = sig * np.sqrt((1 - np.exp(-2 * lam * t)) / (2 * lam))

    mean = A(t, T, lam, sig, ZC) * (1 - beta(t, T, lam) * mu_X_t)
    std = std_X_t * A(t, T, lam, sig, ZC) * beta(t, T, lam)

    return mean, std


def swap_law(t, swap, lam, sig, ZC):
    mu_X_t = 0
    std_X_t = sig * np.sqrt((1 - np.exp(-2 * lam * t)) / (2 * lam))
    if swap.tenor[-2] < t:
        return 0, 0
    else:
        swap_type = (swap.exercice == "payer") * 2 - 1
        maturities = swap.tenor[(swap.tenor < t).sum():]

        deltaK = np.zeros(maturities.shape[0])
        coef = np.zeros(maturities.shape[0])

        coef[0] = 1
        coef[-1] = -1

        deltaK[1:] = np.diff(maturities) * swap.strike

        coef = coef - deltaK

        A_ = A(t, maturities, lam, sig, ZC)
        M = A_ * beta(t, maturities, lam)

        mean = swap_type * swap.nominal * (np.dot(coef, A_) - np.dot(coef, M) * mu_X_t)

        std = swap.nominal * np.dot(coef, M) * std_X_t

        return mean, std


def Mtm_proxy_law(t, portfolio, lam, sig, ZC):
    swaps_law = np.array([swap_law(t, swap, lam, sig, ZC) for swap in portfolio.swaps])
    mean_swaps = swaps_law[:, 0]
    std_swaps = swaps_law[:, 1]

    var = quad_norm(std_swaps ** 2)
    mean = np.sum(mean_swaps)

    return mean, np.sqrt(var)



def Expected_exposure_proxy(t, portfolio, lam, sig, ZC):
    mu, sigma = Mtm_proxy_law(t, portfolio, lam, sig, ZC)
    if sigma == 0:
        return 0
    return ZC.initial_zcb_curve(t) * (mu * norm.cdf(mu / sigma) + sigma * norm.pdf(mu / sigma))


sig_values = np.array([0.005, 0.01, 0.0150, 0.02])
lam_values = np.array([0.01, 0.02, 0.03, 0.04])

# Save the following results
proxy_path = project_path + '\Results\proxy'

# recovery rate & defult probability parameter :
R, lamda = 0.4, 0.005

time_grid = np.linspace(0, T, 500)
PD = np.array([lamda * np.exp(-lamda * t) for t in time_grid])

lam = 0.01
sig_values = np.array([0.01, 0.0150, 0.02])

cva_proxy = np.empty(sig_values.shape[0])
cva_MC = np.empty(sig_values.shape[0])
EE_proxy = np.empty((sig_values.shape[0], 500))
EE_MC = np.empty((sig_values.shape[0], 500))

for sig in sig_values:
    i = list(sig_values).index(sig)

    diff = Diffusion(0, T, n, sig, lam, 60000, pb_measure='Terminal at t')
    print('------------ sig = ', sig)

    tt = time()
    EE = np.array([Expected_exposure_proxy(t, portfolio, lam, sig, ZC) if 0 < t and t < T else 0 for t in time_grid])
    cva_proxy[i] = (np.diff(time_grid) * ((1 - R) * EE * PD)[1:]).sum()
    EE_proxy[i] = EE
    print('proxy done in ', "{}".format(str(timedelta(seconds=round(time() - tt)))))

    tt = time()
    EE = np.array([Expected_exposure_MC(t, portfolio, diff, ZC) if 0 < t and t < T else 0 for t in time_grid])
    cva_MC[i] = (np.diff(time_grid) * ((1 - R) * EE * PD)[1:]).sum()
    EE_MC[i] = EE

    print('MC done in', "{}".format(str(timedelta(seconds=round(time() - tt)))))

    with open(proxy_path + '\Cva_Robustness_test_lam1', 'wb') as f:
        pickle.dump(EE_MC, f)
        pickle.dump(EE_proxy, f)
        pickle.dump(cva_MC, f)
        pickle.dump(cva_proxy, f)











