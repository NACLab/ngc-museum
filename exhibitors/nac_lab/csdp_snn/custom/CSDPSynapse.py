from jax import random, numpy as jnp, jit
from functools import partial
from ngclearn.utils.optim import get_opt_init_fn, get_opt_step_fn
from ngclearn import resolver, Component, Compartment
from ngclearn.components.synapses import DenseSynapse
from ngclearn.utils import tensorstats

class CSDPSynapse(DenseSynapse):

    def __init__(self, name, shape, eta=0., weight_init=None, bias_init=None,
                 w_bound=1., is_nonnegative=False, w_decay=0., update_sign=1.,
                 w_sign=1., is_hollow=False, soft_bound=False, gamma_depress=0.,
                 optim_type="sgd", p_conn=1., resist_scale=1., batch_size=1, **kwargs):
        super().__init__(name, shape, weight_init, bias_init, resist_scale,
                         p_conn, batch_size=batch_size, **kwargs)

        ## synaptic plasticity properties and characteristics
        self.shape = shape
        self.resist_scale = resist_scale
        self.w_bound = w_bound
        self.w_sign = w_sign
        self.w_decay = w_decay ## synaptic decay
        self.eta = eta
        self.gamma_depress = gamma_depress
        self.is_nonnegative = is_nonnegative
        self.is_hollow = is_hollow
        self.soft_bound = soft_bound
        self.update_sign = update_sign

        ## optimization / adjustment properties (given learning dynamics above)
        self.opt = get_opt_step_fn(optim_type, eta=self.eta)

        self.weightMask = 1.
        if self.is_hollow:
            self.weightMask = 1. - jnp.eye(N=shape[0], M=shape[1])

        # compartments (state of the cell, parameters, will be updated through stateless calls)
        self.preVals = jnp.zeros((self.batch_size, shape[0]))
        self.postVals = jnp.zeros((self.batch_size, shape[1]))
        self.preSpike = Compartment(self.preVals)
        self.postSpike = Compartment(self.postVals)
        self.preTrace = Compartment(self.preVals)
        self.postTrace = Compartment(self.postVals)
        self.dWeights = Compartment(jnp.zeros(shape))
        self.dBiases = Compartment(jnp.zeros(shape[1]))

        #key, subkey = random.split(self.key.value)
        self.opt_params = Compartment(get_opt_init_fn(optim_type)(
            [self.weights.value, self.biases.value]
            if bias_init else [self.weights.value]))

    @staticmethod
    def _advance_state(resist_scale, w_sign, inputs, weights, biases):
        factor = w_sign * resist_scale
        outputs = (jnp.matmul(inputs, weights) * factor) + biases
        return outputs

    @resolver(_advance_state)
    def advance_state(self, outputs):
        self.outputs.set(outputs)

    @staticmethod
    def _compute_update(w_bound, update_sign, w_decay, soft_bound, gamma_depress,
                        eta, preSpike, postSpike, preTrace, postTrace, weights,
                        biases):
        ## calculate synaptic update values
        if eta > 0.:
            dW = jnp.matmul(preSpike.T, postTrace) ## pre-syn-driven STDP product
            if gamma_depress > 0.: ## add in post-syn driven counter-term
                dW = dW - jnp.matmul(preTrace.T, postSpike) * gamma_depress
            if soft_bound:
                dW = dW * (w_bound - jnp.abs(weights))
            ## FIXME / NOTE: fix decay to be abs value of synaptic weights
            ## compute post-syn decay factor
            Wdecay_factor = -(jnp.matmul((1. - preSpike).T, (postSpike)) * w_decay)
            db = jnp.sum(postTrace, axis=0, keepdims=True)
        else:
            dW = weights * 0
            Wdecay_factor = dW
            db = biases * 0
        return dW * update_sign, db * update_sign, Wdecay_factor

    @staticmethod
    def _evolve(opt, soft_bound, w_bound, resist_scale, is_nonnegative, update_sign,
                w_decay, bias_init, gamma_depress, is_hollow, eta,
                preSpike, postSpike, preTrace, postTrace, weights, biases, weightMask,
                opt_params):
        d_z = postTrace * resist_scale # 0.1 ## get modulated post-synaptic trace
        ## calculate synaptic update values
        dWeights, dBiases, weightDecay = CSDPSynapse._compute_update(
            w_bound, update_sign, w_decay, soft_bound, gamma_depress, eta, 
            preSpike, postSpike, preTrace, d_z, weights, biases
        )
        ## conduct a step of optimization - get newly evolved synaptic weight value matrix
        if bias_init != None:
            opt_params, [weights, biases] = opt(opt_params, [weights, biases], [dWeights, dBiases])
        else:
            # ignore db since no biases configured
            opt_params, [weights] = opt(opt_params, [weights], [dWeights])
        ## apply decay to synapses and enforce any constraints
        weights = weights + weightDecay
        if w_bound > 0.:
            if is_nonnegative:
                weights = jnp.clip(weights, 0., w_bound)
            else:
                weights = jnp.clip(weights, -w_bound, w_bound)
        if is_hollow: ## enforce lateral hollow matrix masking
            weights = weights * weightMask

        return opt_params, weights, biases, dWeights, dBiases

    @resolver(_evolve)
    def evolve(self, opt_params, weights, biases, dWeights, dBiases):
        self.opt_params.set(opt_params)
        self.weights.set(weights)
        self.biases.set(biases)
        self.dWeights.set(dWeights)
        self.dBiases.set(dBiases)

    @staticmethod
    def _reset(batch_size, shape):
        preVals = jnp.zeros((batch_size, shape[0]))
        postVals = jnp.zeros((batch_size, shape[1]))
        return (
            preVals, # inputs
            postVals, # outputs
            preVals, # pre
            postVals, # post
            jnp.zeros(shape), # dW
            jnp.zeros(shape[1]), # db
        )

    @resolver(_reset)
    def reset(self, inputs, outputs, preSpike, postSpike, preTrace, postTrace,
              dWeights, dBiases):
        self.inputs.set(inputs)
        self.outputs.set(outputs)
        self.preSpike.set(preSpike)
        self.postSpike.set(postSpike)
        self.preTrace.set(preTrace)
        self.postTrace.set(postTrace)
        self.dWeights.set(dWeights)
        self.dBiases.set(dBiases)

    def __repr__(self):
        comps = [varname for varname in dir(self) if Compartment.is_compartment(getattr(self, varname))]
        maxlen = max(len(c) for c in comps) + 5
        lines = f"[{self.__class__.__name__}] PATH: {self.name}\n"
        for c in comps:
            stats = tensorstats(getattr(self, c).value)
            if stats is not None:
                line = [f"{k}: {v}" for k, v in stats.items()]
                line = ", ".join(line)
            else:
                line = "None"
            lines += f"  {f'({c})'.ljust(maxlen)}{line}\n"
        return lines
