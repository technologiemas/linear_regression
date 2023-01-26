import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import numpy.typing as npt
from nptyping import NDArray, Shape, Float
from typing import Any
from statistics import mean, linear_regression
from scipy.stats import linregress, variation, sem, ttest_ind, wald, t
from math import sqrt


"""TYPES"""
NPNum = npt.NDArray[np.float64 | np.int64]  # numpy array with numbers
NP2d = NDArray[Shape['2, 2'], Float]  # 2d numpy array


class Const:
    ALPHA: float = 0  # learning rate
    M: int = 0  # nr of data points
    CYCLES: int = 0


class Nor:
    x_max: float = 0
    x_min: float = 0
    y_max: float = 0
    y_min: float = 0
    y_hat_max: float = 0


# prediction function:
def predict_y(x: NPNum, weight: float, bias: float) -> NPNum:
    """Predicts the y based on given x values, with the weight as the slope and bias as height, as: y = ax + b.
    """
    return weight * x + bias


# loss function:
def calc_loss(y: NPNum, y_hat: NPNum) -> float:
    """
    Calculates the cumulative loss value based on the loss function.
    """
    loss_array = (y_hat - y) ** 2
    loss: float = np.add.reduce(loss_array) * (1 / (2 * Const.M))
    return loss


def plot_regression(x: NPNum, y: NPNum, bias: float) -> None:
    """Plots the calculated formula for the regression line"""
    x_array = [0, max(x)]
    y_array = [bias, max(y)]
    plt.plot(x_array, y_array, '-r')


def plot_data(x: NPNum, y: NPNum, x_label: str, y_label: str) -> None:
    """Plots the data"""
    plt.scatter(x=x, y=y, alpha=0.6)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.ylim(0)
    plt.xlim(0)


def update_weight(weight: float, bias: float, x: NPNum, y: NPNum) -> float:
    der_array = (weight * x + bias - y) * x
    der = np.add.reduce(der_array) * (1 / Const.M)

    weight_new: float = weight - Const.ALPHA * der
    return weight_new


def update_bias(weight: float, bias: float, x: NPNum, y: NPNum) -> float:
    der_array = weight * x + bias - y
    der = np.add.reduce(der_array) * (1 / Const.M)

    bias_new: float = bias - Const.ALPHA * der
    return bias_new


def normalize_data(x: NPNum, y: NPNum, weight: float, bias: float) -> tuple[NPNum, NPNum, float, float]:
    Nor.y_min, Nor.y_max = min(y), max(y)
    Nor.x_min, Nor.x_max = min(x), max(x)
    Nor.y_hat_max = (predict_y(max(x), weight, bias) - Nor.y_min) / (Nor.y_max - Nor.y_min)  # type: ignore

    x_nor = (x - Nor.x_min) / (Nor.x_max - Nor.x_min)
    y_nor = (y - Nor.y_min) / (Nor.y_max - Nor.y_min)
    bias = (bias - Nor.y_min) / (Nor.y_max - Nor.y_min)
    weight = Nor.y_hat_max - bias

    return x_nor, y_nor, weight, bias


def denormalize_data(x_nor: NPNum, y_nor: NPNum, y_hat: NPNum, weight: float, bias: float) -> tuple[NPNum, NPNum, NPNum, float, float]:
    x = x_nor * (Nor.x_max - Nor.x_min) + Nor.x_min
    y = y_nor * (Nor.y_max - Nor.y_min) + Nor.y_min
    y_hat = y_hat * (Nor.y_max - Nor.y_min) + Nor.y_min

    weight = (y_hat[-1] - y_hat[0]) / (x[-1] - x[0])
    bias = y_hat[-1] - weight * max(x)

    # loss = loss * (Nor.y_max ** 2 - Nor.y_min ** 2) + Nor.y_min ** 2

    return x, y, y_hat, weight, bias


def get_df(path: str, dep: list[str], ind: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path) \
        .sort_values(*ind) \
        .dropna(subset=[*dep, *ind])
    return df


def calc_loss_vectorized(x: NPNum, y: NPNum, weights: NP2d, biases: NP2d) -> tuple[NP2d, NP2d]:
    """
    Inputs weights and biases 2d arrays filled with possible values for weight and bias.
    Calculates using vectorization.
    Returns a 2d np array shaped (len(weights), len(biases[0])) with a loss value calculated for
    each weight and bias that are given.
    """

    losses: NP2d = np.ndarray((len(weights), len(biases[0])), dtype=float)
    y_hats: NP2d = np.ndarray((len(weights), len(biases[0])), dtype=float)
    # losses.fill(0)  # for debugging

    biases = [i[0] for i in biases]
    weights = weights[0]

    for row, bias in enumerate(biases):
        for col, weight in enumerate(weights):
            y_hat = weight * x + bias

            loss_array: NPNum = (y_hat - y) ** 2
            loss: float = np.add.reduce(loss_array) * (1 / (2 * Const.M))

            losses[row][col] = loss
            y_hats[row][col] = y_hat[-1]

    # losses =

    return losses, y_hats


def denormalize_gradient_descent(x: NPNum, y_hats: NP2d, weights: NP2d, biases: NP2d) -> tuple[NP2d, NP2d]:
    x = x * (Nor.x_max - Nor.x_min) + Nor.x_min
    y_hats = y_hats * (Nor.y_max - Nor.y_min) + Nor.y_min
    biases_denorm = biases * (Nor.y_max - Nor.y_min) + Nor.y_min
    weights_denorm = (y_hats - biases_denorm) / max(x)

    weights = weights_denorm
    biases = biases_denorm

    return weights, biases


def gradient_descent(x: NPNum, y: NPNum, y_hat: NPNum) -> tuple[NP2d, NP2d, NP2d]:
    # x = np.array([1, 2, 3, 4, 5, 6])
    # y = np.array([1, 2, 3, 4, 5, 6])

    delta_w: float = float(np.max(x) / 10)  # amount of steps to plot for weight
    delta_b: float = delta_w
    min_w, max_w = -4 * max(x), 4 * max(x)  # max slope of 4*x
    min_b, max_b = min(y) - abs(max(y) - min(y)), max(y) + abs(max(y) - min(y))  # max b of 2 * y range

    weights: NP2d = np.arange(min_w, max_w, delta_w, dtype=float)  # pre-allocate arrays for possible weights and biases
    biases: NP2d = np.arange(min_b, max_b, delta_b, dtype=float)
    weights, biases = np.meshgrid(weights, biases)  # create a 2d mesh grid of all possible weights and biases

    losses, y_hats = calc_loss_vectorized(x, y, weights, biases)
    losses = np.log10(losses, out=np.zeros_like(losses), where=(losses != 0))  # take the log for clearer plot

    weights, biases = denormalize_gradient_descent(x, y_hats, weights, biases)

    return weights, biases, losses


def plot_gradient_descent(weights_mesh: NP2d, biases_mesh: NP2d, losses_mesh: NP2d) -> None:
    ax = plt.figure().add_subplot(projection='3d')
    # plt.plot(weights, biases, loss, '-r')
    ax.plot_surface(weights_mesh, biases_mesh, losses_mesh, alpha=0.9)
    ax.contourf(weights_mesh, biases_mesh, losses_mesh, levels=20, zdir='z', offset=np.min(losses_mesh), cmap='coolwarm')
    ax.set(xlabel='weight', ylabel='bias', zlabel='log loss', alpha=.6)


def update_parms(weight: float, bias: float, x: NPNum, y: NPNum) -> tuple[float, float]:
    weight_new = update_weight(weight, bias, x, y)
    bias_new = update_bias(weight, bias, x, y)
    return weight_new, bias_new


def calc_r_squared(y: NPNum, y_hat: NPNum) -> float:
    sst = np.add.reduce((y-mean(y))**2)
    ssr = np.add.reduce((y_hat-mean(y))**2)

    r_squared = ssr/sst
    return r_squared


def print_statistics(r_squared, weight, bias, p) -> None:  # type: ignore
    print("Weight: ", weight)
    print("Bias: ", bias)
    # r = sqrt(r_squared)
    print("R^2: ", r_squared)
    print("p-value: ", p)


def calc_p_value(y_hat: NPNum, y: NPNum, x: NPNum, weight: float) -> float:
    dof = Const.M - 2
    sse = np.add.reduce((y_hat - y) ** 2)
    x_sst = np.add.reduce((x - mean(x)) ** 2)
    se = sqrt((1 / dof) * (sse / x_sst))
    t_val = weight / se
    p_value: float = t.sf(abs(t_val), dof) * 2  # *2 for two-tailed p-value

    return p_value


def settings_plot():  # type: ignore
    plt.xlabel("area (m²)")
    plt.ylabel("price")
    plt.title("Random line (weight 1000, bias: 500000)")
    plt.tight_layout()


def run_regression(weight: float, bias: float, x: NPNum, y: NPNum) -> tuple[float, float, NPNum, NPNum]:
    assert all(len(i) == len(y) for i in iter([x, y]))  # assert all arrays of variables are of equal length
    loss: NPNum = np.array([])
    y_hat: NPNum = np.array([])

    for _ in range(Const.CYCLES):
        weight, bias = update_parms(weight, bias, x, y)

        y_hat = predict_y(x, weight, bias)
        loss = np.append(loss, calc_loss(y, y_hat))

    return weight, bias, y_hat, loss


def calc_stats(x, y, y_hat, weight, bias):  # type: ignore
    r_squared = calc_r_squared(y, y_hat)
    p_value = calc_p_value(y_hat, y, x, weight)
    print_statistics(r_squared, weight, bias, p_value)
    a = linregress(x, y)  # test with real regression
    print()
    print(a)
    print("Scipy.stats r-squared: ", a.rvalue ** 2)
    print("Scipy.stats p-value: ", a.pvalue)


def main():  # type: ignore
    """
    Note: The data has to be ordered based on x (ind) values.
    """
    path: str = "dataset/Housing.csv"
    dep_str: list[str] = ['price']
    ind_str: list[str] = ['area']

    # initializing values:
    df: pd.DataFrame = get_df(path, dep_str, ind_str)  # sort based on x values!
    weight: float = 950
    bias: float = 519000
    y_hats: NPNum = np.array([])
    ind: list[NPNum] = [df[i].to_numpy() for i in ind_str]  # initialize array for x values
    x: NPNum = ind[0]
    y: NPNum = df[dep_str[0]].to_numpy()
    Const.M = len(y)
    Const.ALPHA = 0.1
    Const.CYCLES = 5000

    x = x / 100  # to get area in m^2
    y = y / 10  # to get currency adjusted for inflation to modern prices (as this is an older dataset)

    x, y, weight, bias = normalize_data(x, y, weight, bias)  # normalizes the data to a [0, 1] range, min-max normalization

    weight, bias, y_hat, loss = run_regression(weight, bias, x, y)

    weights_mesh, biases_mesh, losses_mesh = gradient_descent(x, y, y_hat)

    x, y, y_hat, weight, bias = denormalize_data(x, y, y_hat, weight, bias)  # weight and bias not correct yet

    # uncomment the type of graph you want to plot:
    plot_data(x=x, y=y, x_label=ind_str[0], y_label=dep_str[0])
    plot_regression(x, y_hat, bias)
    # plot_gradient_descent(weights_mesh, biases_mesh, losses_mesh)
    # plt.plot(loss)

    settings_plot()  # type: ignore
    # plt.savefig("random_line", dpi=300)  # uncomment to save figure
    plt.show()

    calc_stats(x, y, y_hat, weight, bias)  # type: ignore


main()  # type: ignore
