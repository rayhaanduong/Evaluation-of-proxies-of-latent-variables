import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt



def get_A_M_N_R(X, G, r):
    
    """
    Compute A(j), M(j), NS(j), and R²(j) statistics for observed factors G.

    Parameters
    ----------
    X : ndarray of shape (T, N)
        Standardized data matrix for N cross-sectional units over T time periods.
        Each column corresponds to one asset/variable.
    
    G : ndarray of shape (T, m)
        Matrix of observed factors (proxies) for m variables over T time periods.
    
    r : int
        Number of latent factors to estimate via principal components.

    Returns
    -------
    A : ndarray of shape (m,)
        Frequency of exceeding the critical value for each observed factor G_j.

    M : ndarray of shape (m,)
        Maximum t-statistic over time for each G_j.

    NS : ndarray of shape (m,)
        Noise-to-signal ratio for each observed factor.

    R2 : ndarray of shape (m,)
        Explained variance ratio for each observed factor.
    """
    
    
    #We normalized the quantities
    
    X_mean = X.mean(axis = 0)
    X_std = X.std(axis=0, ddof = 1)
    X_standardized = (X - X_mean) / X_std
    
    G_mean = G.mean(axis=0)
    G_std = G.std(axis=0, ddof=1)
    G_standardized = (G - G_mean) / G_std
    
    T = X.shape[0]
    N = X.shape[1]
    m = G.shape[1]
    
    #ACP
    
    cov_X = X_standardized@X_standardized.T / (N*T)
    eigvals, U = np.linalg.eigh(cov_X)

    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    U = U[:, idx]
    
    F_hat = np.sqrt(T) * U[:,:r]
    Lambda_hat = (X_standardized.T @ F_hat) / T
    
    
    #Test
    
    t_stats = np.zeros((T, m))
    d_hat = np.zeros((r, m))   
    G_hat = np.zeros((T, m)) 
    
    
    A = np.zeros(m)
    M = np.zeros(m)
    
    NS = np.zeros(m)
    R2 = np.zeros(m)



    for j in range(m):
        G_j = G_standardized[:,j]
        d_hat[:, j] = np.linalg.inv(F_hat.T @ F_hat) @ (F_hat.T @ G_j)
        G_hat[:, j] = F_hat @ d_hat[:, j]
        var_G_hat = np.var(G_hat[:, j], ddof=1)
        t_stats[:, j] = (G_hat[:, j] - G_j) / np.sqrt(var_G_hat)


    alpha = 0.025
    F_alpha = norm.ppf(1 - alpha/2)

    for j in range(m):
        A[j] = np.mean(np.abs(t_stats[:, j]) > F_alpha)
        M[j] = np.max(np.abs(t_stats[:, j]))

        dj_hat = np.linalg.lstsq(F_hat, G_standardized[:, j], rcond=None)[0]
        G_proj = F_hat @ dj_hat  
        error = G_standardized[:, j] - G_proj
        
        var_error = np.var(error, ddof=1)
        var_proj = np.var(G_proj, ddof=1)
        var_total = np.var(G_standardized[:, j], ddof=1)
        
        NS[j] = var_error / var_proj
        R2[j] = var_proj / var_total
        
        
    j_indices = np.arange(1, m+1)  

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # A(j) : frequency of exceeding the critical values
    axs[0, 0].bar(j_indices, A)
    axs[0, 0].set_title("A(j) statistic")
    axs[0, 0].set_xlabel("G_j")
    axs[0, 0].set_ylabel("Fréquence")

    # M(j) : max t-statistic
    axs[0, 1].bar(j_indices, M)
    axs[0, 1].set_title("M(j) statistic")
    axs[0, 1].set_xlabel("G_j")
    axs[0, 1].set_ylabel("Max t-stat")

    # NS(j) : Noise-to-signal
    axs[1, 0].bar(j_indices, NS)
    axs[1, 0].set_title("NS(j) statistic")
    axs[1, 0].set_xlabel("G_j")
    axs[1, 0].set_ylabel("Noise / Signal")

    # R²(j) : Explained variance
    axs[1, 1].bar(j_indices, R2)
    axs[1, 1].set_title("R²(j) statistic")
    axs[1, 1].set_xlabel("G_j")
    axs[1, 1].set_ylabel("R²")

    plt.tight_layout()
    plt.show()
    
    
    return A, M, NS, R2