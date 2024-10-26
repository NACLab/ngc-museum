import pandas as pd
from ngclearn import Context, numpy as jnp
from typing import Optional
import matplotlib.pyplot as plt


class Std_SINDy():
    '''
    STLSQ (Sequential Thresholded Least Squares)
    '''
    def __init__(self, threshold=0.5, max_iter=20):
        self.library_type: str = 'polynomial'
        self.polynomial_order: int = 2
        self.threshold: float = threshold
        self.max_iter: int = max_iter
        self.coef_:Optional[jnp.ndarray] = None
        self.lib_:Optional[jnp.ndarray] = None


    def fit(self, dx: jnp.ndarray, lib: jnp.ndarray) -> jnp.ndarray:
        # Solve for coef that gives min ||lib @ coef - dx||^2
        coef = jnp.linalg.lstsq(lib, dx, rcond=None)[0]

        for iter_ in range(self.max_iter):
            coef_pre = jnp.array(coef)
            coef_zero = jnp.zeros_like(coef)

            res_idx = jnp.where(jnp.abs(coef) >= self.threshold, True, False)
            res_mask = jnp.any(res_idx, axis=1)
            res_lib = lib[:, res_mask]

            coef_new = jnp.linalg.lstsq(res_lib, dx, rcond=None)[0]
            sparse_coef = coef_zero.at[res_mask].set(coef_new)

            coef = jnp.where(jnp.abs(sparse_coef) >= self.threshold, sparse_coef, 0.)

            if jnp.allclose(sparse_coef, coef_pre, rtol=1e-4, atol=1e-4):
                # print('converget at iteration = {}'.format(iter_))
                break

        self.coef_ = coef
        self.lib_ = lib
        return coef

    def predict(self) -> jnp.ndarray:
        '''
            Predict time derivatives using the learned SINDy model.

        Returns
        -------
        dx_pred : jnp.ndarray
        '''

        return self.lib_ @ self.coef_

    def get_ode(self, feature_names):
        c_bool = jnp.where(self.coef_ == 0, False, True)
        idx_ = jnp.any(c_bool == True, axis=1)
        c_ = list(self.coef_[idx_])
        n_ = [res_trm for res_trm, i_ in zip(feature_names, idx_) if i_]
        visual_eq = [f"{c_[i][0]:.3f} " + n_[i] + (' +' if (i != len(c_)-1) and (c_[i+1][0]>0) else ' ') for i in range(len(c_))]

        return visual_eq

    def vis_sys(self, ts, dX, pred, model):

        plt.figure(facecolor='floralwhite')

        plt.plot(ts, dX[:, 0], label=r'$\dot{x}$', linewidth=5, alpha=0.3, color='turquoise')
        plt.plot(ts, pred[:, 0], label=r'$\hat{\dot{x}}$', linewidth=0.8, ls="--", color='black')

        if dX.shape[1] >= 2:
            plt.plot(ts, dX[:, 1], label=r'$\dot{y}$', linewidth=4, alpha=0.6, color='pink')
            plt.plot(ts, pred[:, 1], label=r'$\hat{\dot{y}}$', linewidth=0.8, ls='--', color='red')

        if dX.shape[1] >= 3:
            plt.plot(ts, dX[:, 2], label=r'$\dot{z}$', linewidth=2, alpha=0.8, color='yellow')
            plt.plot(ts, pred[:, 2], label=r'$\hat{\dot{z}}$', linewidth=0.8, ls="--", color='navy')

        plt.grid(True)
        plt.legend(loc='lower right')
        plt.xlabel(r'$time$', fontsize=10)
        plt.ylabel(r'$\{\dot{x}, \dot{y}, \dot{z}\}$', fontsize=8)

        plt.title('Sparse Coefficients of "{}" model'.format(model.__name__))
        plt.show()

