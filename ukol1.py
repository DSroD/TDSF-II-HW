#.___________. _______       _______. _______                __    __  ____    __    ____  __  
#|           ||       \     /       ||   ____|              |  |  |  | \   \  /  \  /   / /_ | 
#`---|  |----`|  .--.  |   |   (----`|  |__       ______    |  |__|  |  \   \/    \/   /   | | 
#    |  |     |  |  |  |    \   \    |   __|     |______|   |   __   |   \            /    | | 
#    |  |     |  '--'  |.----)   |   |  |                   |  |  |  |    \    /\    /     | | 
#    |__|     |_______/ |_______/    |__|                   |__|  |__|     \__/  \__/      |_| 
#                                                                                                                                                                                     

#%% [markdown]
# Stačí spustit celý skript v pythonu (psáno ve verzi 3.8.1, mělo by fungovat na >3.6), vygeneruje všechny soubory.
#
# Použté balíčky (nutné pro běh) - numpy, scipy, matplotlib.
#
# Dá se spustit i bez balíčku progressbar2, nutno ale upravit 1 for cyklus ve funkci 'vysece'.
# Balíček progressbar se instaluje pomocí 'pip install --user progressbar2'.
# Pokud nemáte balíček plotly, zákomentujte na úplném konci kódu (v main části) poslední dva příkazy, možná bude potřeba
# zakomentovat i definici dekorátoru 'plotly_3d_plot' a použitý dekorátor ve 'funkci gen_sph'.
# Balíček plotly se instaluje pomocí 'pip install --user plotly'.
#
# Prvních cca 220 řádků jsou pouze definice funkcí a dekorátorů, procedurální kód je až na posledních cca 15 řádcích.
#
# Pokud vaše IDE podporuje Interactive mode, je kód rozdělený na celly - 1. cell jsou definice funkcí, zbytek jsou jednotlivé úlohy,
# dekorátory poskytující vizualizaci (plot dekorátory) obsahují .show(), takže se v interactive módu zobrazí.

#%%
import numpy as np
import matplotlib.pyplot as plt
import progressbar  # 'pip install --user progressbar2'   (nebo ve funkci 'vysece' zakomentovat for loop s progressbarem a použít zakomentovaný progressbar)

# Pokud nemáte plotly, zakomentujte na úplném konci kódu poslední dva příkazy, případně 'pip install --user plotly'

def plot_scatter(func):
    def wrapper(emin, emax, num, **kwargs):
        title = kwargs.get('title', "Scatter plot")
        x, y = func(emin, emax, num, **kwargs)
        fig = plt.figure(figsize=(8,8))
        plt.scatter(x, y, s=3)
        plt.xlabel(u"{}".format(kwargs.get('xlabel', u"Počet vzorků")))
        plt.ylabel(u"{}".format(kwargs.get('ylabel', u"$\sigma^2$")))
        plt.title(title)
        plt.gca().set_yscale('log')
        plt.gca().set_xscale('log')
        plt.show()
        return (x, y), fig
    return wrapper

def plotly_plot_sphere(func):
    import plotly.graph_objects as go
    def wrapper(N, **kwargs):
        u, v = np.mgrid[0:2*np.pi:100j, 0:np.pi:100j]
        x = np.sin(u) * np.cos(v)
        y = np.sin(u) * np.sin(v)
        z = np.cos(u)
        fig = go.Figure(data=[go.Surface(x=x, y=y, z=z)])
        fig.show()
    return wrapper

def plotly_3d_plot(func):
    import plotly.graph_objects as go
    def wrapper(N, **kwargs):
        vec = func(N)
        fig = go.Figure(data=[go.Scatter3d(x=vec[0], y=vec[1], z=vec[2], mode='markers', marker=dict(size=3, line=dict(width=1)))])
        fig.show()
        fig.write_html("vec_{}.html".format(N))
    return wrapper

        
def plot_hist_fit(func):
    def gauss(x):
        return 1/np.sqrt(2 * np.pi) * np.exp(-1/2 * x*x)
    def wrapper(N, **kwargs):
        left = kwargs.get('left', -2)
        right = kwargs.get('right', 2)
        color = kwargs.get('boxcolor', None)
        edgecolor = kwargs.get('edgecolor', None)
        title = kwargs.get('title', "{} vygenerovaných čísel")
        bins = np.arange(left, right, kwargs.get('bin_spacing', 0.05))
        y = func(N, **kwargs)
        fig = plt.figure(figsize=(8,8))
        plt.hist(y, bins=bins, density=kwargs.get('normed', True), color=color, edgecolor=edgecolor)
        plt.plot(bins, gauss(bins), "k--")
        plt.xlabel(u"$x$")
        plt.ylabel(u"$n$")
        plt.title(title.format(N))
        plt.show()
        return y, fig
    return wrapper

def savefig(name):
    numb = {}
    def deco(func):
        def wrapper(*args, **kwargs):
            nu = numb.get(name, 0)
            numb[name] = nu + 1
            x, fig = func(*args, **kwargs)
            fig.savefig("{}_{}.pdf".format(name, nu), dpi=400)
            return x
        return wrapper
    return deco

def cauchy_cdf_inverse(x, gamma=1):
    return gamma * np.tan(np.pi * (x + 1/2))

from scipy.special import erfinv

def gauss_cdf_inverse(x):
    return np.sqrt(2) * erfinv(2*x-1)

def to_cart(theta: float, phi: float) -> np.ndarray:
    return np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])



@savefig("cauchy")
@plot_scatter
def variance(emin: int, emax: int, num: int, gamma: int=1, **kwargs) -> tuple:
    """Comutes variances of different sized Cauchy-distributed (comuted using inverse CDF, centered) numbers.

    Args:
        emin (int): 10^emin is lower bound of size of generated data.
        emax (int): 10^emax is upper bound of size of generated data.
        num (int): Number of generated variances (number of points between 10^emin and 10^emax).
        gamma (int, optional): Gamma parameter of Cauchy distribution. Defaults to 1.

    Returns:
        tuple: x, y, points in graph
    """
    
    x = [int(x//1) for x in np.logspace(emin, emax, num=num)]
    y = [np.var(cauchy_cdf_inverse(np.random.uniform(0,1,n))) for n in x]
    return x, y

@savefig("gauss")
@plot_hist_fit
def gauss_cdfinv(N: int, **kwargs) -> np.ndarray:
    """Generate N random numbers using inverse CDF.

    Args:
        N (int): Number of numbers to generate.

    Returns:
        np.ndarray: Array of normal-distributed numbers.
    """
    return gauss_cdf_inverse(np.random.uniform(0,1,N))

@savefig("gauss_box")
@plot_hist_fit
def gauss_boxmuller(N: int, **kwargs) -> np.ndarray:
    """Generate N random numbers using Box-Muller method.

    Args:
        N (int): Number of numbers to generate.

    Returns:
        np.ndarray: Array of normal-distributed numbers.
    """
    U1 = np.random.uniform(0,1,N)
    U2 = np.random.uniform(0,1,N)
    return np.sqrt(-2 * np.log(U1)) * np.cos(2*np.pi * U2)

def randsphere(N: int) -> np.ndarray:
    """Generates N random angles such that unit vectors pointing along generated vectors
       are distributed equally on 2-sphere.

    Args:
        N (int): Number of vectors to generate.

    Returns:
        np.ndarray: Array of theta, phi angles.
    """

    # thetu musím přetransformovat žejo
    return np.arccos(1 - 2 * np.random.uniform(0,1,N)), np.random.uniform(0,1,N) * 2 * np.pi

@plotly_3d_plot
def gen_sph(N:int) -> np.ndarray:
    """Generate N random 3-dimensional cartesian unit vectors distributed equally on 2-sphere.

    Args:
        N (int): Number of vectors to generate.

    Returns:
        np.ndarray: 3 row N column matrix consisting of x, y, z values of random vectors.
    """
    return to_cart(*randsphere(N))

@savefig("sphere")
@plot_scatter
def vysece(emin: int, emax: int, num: int, **kwargs) -> tuple:
    """Compare number of random vectors in 2 spherical sectors and plot graph.

    Args:
        emin (int): 10^emin is lower bound of size of generated data.
        emax (int): 10^emax is upper bound of size of generated data.
        num (int): number of simulations (number of points between 10^emin and 10^emax).

    Returns:
        tuple: x, y points in graph
    """
    xs = []
    pom = []
    for N, _ in zip(np.logspace(emin, emax, num=num), progressbar.progressbar(range(num))):
    #for N, in np.logspace(exp0, exp1, num=num):
        N = int(N//1)
        theta, phi = randsphere(N)
        siz = 1 - 3/16 # V = 2/3 * \pi * (1-cos(\phi)); S_sph = 4 * \pi
        v1 = [to_cart(t, p) for t, p in zip(theta, phi) if np.dot(to_cart(t,p), np.array([1,0,0])) > siz]
        v2 = [to_cart(t, p) for t, p in zip(theta, phi) if np.dot(to_cart(t,p), 1/np.sqrt(3) * np.array([1,1,1])) > siz]
        if len(v1) != 0 and len(v2) != 0:
            pom.append(len(v1) / len(v2))
            xs.append(N)
    return xs, pom

#%%
if __name__ == '__main__':

#%%
    # Variance (a) --------------
    variance(4, 6, 800, title="Variance v závislosti na počtu čísel") # Chvilku běží
#%%
    # Gauss (b) -----------------
    gauss_cdfinv(100000, bin_spacing=0.05, normed=True, left=-3, right=3, edgecolor="black", title="{} vygenerovaných čísel pomocí inverse CDF")
    gauss_cdfinv(1000, bin_spacing=0.05, normed=True, left=-5, right=5, edgecolor="black", title="{} vygenerovaných čísel pomocí inverse CDF")
    gauss_boxmuller(100000, bin_spacing=0.05, normed=True, left=-3, right=3, boxcolor="orange", edgecolor="black", title=u"{} vygenerovaných čísel pomocí Box-Mullerovy metody")
    gauss_boxmuller(1000, bin_spacing=0.05, normed=True, left=-5, right=5, boxcolor="orange", edgecolor="black", title=u"{} vygenerovaných čísel pomocí Box-Mullerovy metody")
#%%
    # Výseče (c) - trvají poměrně dlouho, proto jsem použil progressbar2, ať je vidět jak dlouho ještě!
    vysece(3, 5, 200, xlabel="Počet vektorů", ylabel="Poměr protnutí výsečí", title="Generování výsečí")
    gen_sph(1000)    # zakomentujte, pokud nemáte plotly
    gen_sph(10000)   # zakomentujte, pokud nemáte plotly


# %%
# ██████████ ██████   █████       █████    ███████    █████ █████
#░░███░░░░░█░░██████ ░░███       ░░███   ███░░░░░███ ░░███ ░░███ 
# ░███  █ ░  ░███░███ ░███        ░███  ███     ░░███ ░░███ ███  
# ░██████    ░███░░███░███        ░███ ░███      ░███  ░░█████   
# ░███░░█    ░███ ░░██████        ░███ ░███      ░███   ░░███    
# ░███ ░   █ ░███  ░░█████  ███   ░███ ░░███     ███     ░███    
# ██████████ █████  ░░█████░░████████   ░░░███████░      █████   
#░░░░░░░░░░ ░░░░░    ░░░░░  ░░░░░░░░      ░░░░░░░       ░░░░░                                                      
#
#
#                      ▄              ▄    
#                     ▌▒█           ▄▀▒▌   
#                     ▌▒▒█        ▄▀▒▒▒▐   
#                    ▐▄█▒▒▀▀▀▀▄▄▄▀▒▒▒▒▒▐   
#                  ▄▄▀▒▒▒▒▒▒▒▒▒▒▒█▒▒▄█▒▐   
#                ▄▀▒▒▒░░░▒▒▒░░░▒▒▒▀██▀▒▌   
#               ▐▒▒▒▄▄▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▀▄▒▌  
#               ▌░░▌█▀▒▒▒▒▒▄▀█▄▒▒▒▒▒▒▒█▒▐  
#              ▐░░░▒▒▒▒▒▒▒▒▌██▀▒▒░░░▒▒▒▀▄▌ 
#              ▌░▒▒▒▒▒▒▒▒▒▒▒▒▒▒░░░░░░▒▒▒▒▌ 
#             ▌▒▒▒▄██▄▒▒▒▒▒▒▒▒░░░░░░░░▒▒▒▐ 
#             ▐▒▒▐▄█▄█▌▒▒▒▒▒▒▒▒▒▒░▒░▒░▒▒▒▒▌
#             ▐▒▒▐▀▐▀▒▒▒▒▒▒▒▒▒▒▒▒▒░▒░▒░▒▒▐ 
#              ▌▒▒▀▄▄▄▄▄▄▒▒▒▒▒▒▒▒░▒░▒░▒▒▒▌ 
#              ▐▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░▒░▒▒▄▒▒▐  
#               ▀▄▒▒▒▒▒▒▒▒▒▒▒▒▒░▒░▒▄▒▒▒▒▌  
#                 ▀▄▒▒▒▒▒▒▒▒▒▒▄▄▄▀▒▒▒▒▄▀   
#                   ▀▄▄▄▄▄▄▀▀▀▒▒▒▒▒▄▄▀     
#                      ▀▀▀▀▀▀▀▀▀▀▀▀        

# %%
