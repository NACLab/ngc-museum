import pandas as pd
from ngclearn import Context, numpy as jnp
from typing import Optional


class Std_SINDy():
    '''
    An ngc-learn implementation/reproduction of the Sparse Identification of Non-linear Dynamical systems 
    (SINDy) model proposed in: 

    Brunton, Steven L., Joshua L. Proctor, and J. Nathan Kutz. "Discovering governing equations 
    from data by sparse identification of nonlinear dynamical systems." Proceedings of the 
    national academy of sciences 113.15 (2016): 3932-3937.
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

        for _ in range(self.max_iter):
            coef_pre = coef.copy()

            coef_zero = jnp.zeros_like(coef)

            res_idx = jnp.where(jnp.abs(coef) >= self.threshold, True, False)
            res_mask = jnp.any(res_idx, axis=1)
            res_lib = lib[:, res_mask]

            # repeating least square with residual features in libreary
            coef_new = jnp.linalg.lstsq(res_lib, dx, rcond=None)[0]
            sparse_coef = coef_zero.at[res_mask].set(coef_new)

            # remove non-sharing coefficients for each dims in feature librate
            coef = jnp.where(jnp.abs(sparse_coef) >= self.threshold, coef, 0.)

            # convergence check
            if jnp.allclose(coef, coef_pre, rtol=1e-6, atol=1e-6):
                print('Model converged at iteration = {}'.format(_))
                break ## engage early stopping if condition met

        self.coef_ = coef
        self.lib_ = lib
        self.dx_ = dx
        return coef

    def predict(self) -> jnp.ndarray:
        '''
            Predict the time derivatives using the learned SINDy model.

            Returns:
                dx_pred : jnp.ndarray
        '''

        return self.lib_ @ self.coef_


    def error(self) -> jnp.ndarray:
        '''
            Frobenius norm for prediction and target difference

            Returns:
                error (between model prediction and correct derivative)
        '''

        return jnp.linalg.norm(self.predict() - self.dx_)

    def get_ode(self, feature_names):

        # print coeff and plot
        c_bool = jnp.where(self.coef_ == 0, False, True)
        idx_ = jnp.any(c_bool == True, axis=1)
        c_ = self.coef_[idx_]
        n_ = [res_trm for res_trm, i_ in zip(feature_names, idx_) if i_]

        if self.coef_.shape[1] == 3:
            df_ = pd.DataFrame(jnp.round(c_, 3), index=n_, columns=['ẋ', 'ẏ', 'ż'])

        elif self.coef_.shape[1] == 2:
            df_ = pd.DataFrame(jnp.round(c_, 3), index=n_, columns=['ẋ', 'ẏ'])

        model_ode = df_
        return model_ode
