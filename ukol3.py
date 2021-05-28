#(`-')     _(`-')    (`-').->                                               
#( OO).-> ( (OO ).-> ( OO)_    <-.                           .->            
#/    '._  \    .'_ (_)--\_)(`-')-----.     (`-')       ,--.(,--.   .----.  
#|'--...__)'`'-..__)/    _ /(OO|(_\---'     ( OO).->    |  | |(`-')\_.-,  | 
#`--.  .--'|  |  ' |\_..`--. / |  '--.     (,------.    |  | |(OO )  |_  <  
#   |  |   |  |  / :.-._)   \\_)  .--'      `------'    |  | | |  \.-. \  | 
#   |  |   |  '-'  /\       / `|  |_)                   \  '-'(_ .'\ `-'  / 
#   `--'   `------'  `-----'   `--'                      `-----'    `---''  


from typing import Tuple, Sequence
import numpy as np
import matplotlib.pyplot as plt

mu = 1
a = 1
h = 1

def savefig(name : str):
    numb = {}
    def deco(func):
        def wrapper(*args, **kwargs):
            nu = numb.get(name, 0)
            numb[name] = nu + 1
            x, y, fig = func(*args, **kwargs)
            fig.savefig("{}_{}.png".format(name, nu), dpi=4000)
            return x, y
        return wrapper
    return deco

def plot_multiple(func):
    def deco(*args, **kwargs) -> Tuple[Sequence, Sequence, plt.Figure]:
        ret = func(*args, **kwargs)
        x = ret[0]
        fig = plt.figure(figsize=kwargs.get('figsize', (8,8)))
        for index, y in enumerate(ret[1:]):
            plt.plot(x, y, label=kwargs.get('legend', [f"data {i+1}" for i in range(len(ret[1:]))])[index])
        plt.legend()
        plt.xlabel(u"{}".format(kwargs.get('xlabel', u"$x$")))
        plt.ylabel(u"{}".format(kwargs.get('ylabel', u"$y$")))
        plt.title(u"{}".format(kwargs.get('title', u"Graph")))
        
        if kwargs.get('scale', '') == 'log':
            plt.gca().set_yscale('log')
            plt.gca().set_xscale('log')
        plt.show()
        return x, y, fig
    return deco

def numerical_sum(N : float):
    def num_sum(func):
        def deco(x : float, **kwargs) -> float:
            sn = func(0, x)
            for j in range(1, N):
                sn += func(j, x)
            return sn
        return deco
    return num_sum

def numerical_sum_eps(eps : float):
    def num_sum(func):
        def deco(x : float, **kwargs) -> float:
            s, sn = func(0, x), func(0, x)
            sl = sn + 2*eps
            j = 1
            while np.all((sn - sl) > eps):
                sn = func(j, x)
                s += sn
                sl = sn
                j += 1
            return s
        return deco
    return num_sum

def numerical_derivative(eps: float):
    def num_deriv(func):
        def deco(x : float, **kwargs) -> float:
            dx = (-func(x + 2*eps) + 8*func(x + eps) - 8*func(x - eps) + func(x - 2*eps)) / (12 * eps)
            return dx
        return deco
    return num_deriv


def e(j : int) -> float:
   return j * (j+1) * h * h / (2 * mu * a*a)

def g(j : int) -> float:
    return 2 * j + 1


@numerical_sum(50)
def parsumT(j, T):
    return g(j) * np.exp(- e(j)/T)


@numerical_sum(50)
def deriv_parsumT(j, T):
    return g(j) * e(j) * np.exp(- e(j)/T)


@numerical_sum(2)
def parsumT_low(j, T):
    return g(j) * np.exp(- e(j)/T)


@numerical_sum(2)
def deriv_parsumT_low(j, T):
    return g(j) * e(j) * np.exp(- e(j)/T)


@savefig("mean_energy_T")
@plot_multiple
def mean_enT(*args, **kwargs):
    return args[0], deriv_parsumT(args[0]) /  parsumT(args[0]), deriv_parsumT_low(args[0]) / parsumT_low(args[0]), args[0]


@numerical_derivative(0.00001)
def capacity(beta):
    return deriv_parsumT(beta) / parsumT(beta)

@numerical_derivative(0.00001)
def capacity_low(beta):
    return deriv_parsumT_low(beta) / parsumT_low(beta)

@savefig("capacity")
@plot_multiple
def cap(*args, **kwargs):
    return args[0], capacity(args[0]), capacity_low(args[0]), [1 for _ in args[0]]


if __name__ == "__main__":
    x1, en = mean_enT(np.linspace(0.001, 1.2, num=1000,), xlabel=u"$T$", ylabel="$E$", title="Mean energy", legend=["Numerical", "Low", "High"])
    x2, en = cap(np.linspace(0.001, 1.2, num=1000), xlabel=u"$T$", ylabel="$C$", title="Capacity", legend=["Numerical", "Low", "High"])