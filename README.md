[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/downloads)[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity)

# NGC-Museum: Neuroscience Models and NeuroAI Agents with NGC-Learn

<b>ngc-museum</b> is a public repository for <i><a href="https://github.com/NACLab/ngc-learn/">ngc-learn</a></i> that houses biomimetic, brain-inspired computing, and computational neuroscience / biophysics models proposed throughout history. All models in this repo, whether contributed by community, other groups, or the ngc-learn dev team, are written in Python using ngc-learn (and JAX). Each model in the `exhibits/` directory or collection of models in the `exhibitors/` sub-directories contain `README` top-level files that explain their central properties and general organization of the sub-directory they are found within, including model/agent simulation instructions, problem task descriptions, as well as relevant hyper-parameter values need to reproduce experimental results.

For official walkthroughs going over the model exhibits found in this repo, please visit the ngc-learn documentation page: https://ngc-learn.readthedocs.io/ (under the "<a href="https://ngc-learn.readthedocs.io/en/latest/museum/model_museum.html">Model Museum</a>" side-bar). For information, including anything related to usage instructions and details related to ngc-learn itself, please refer to the official ngc-learn <a href="https://github.com/NACLab/ngc-learn/">repo</a> (and its <a href="https://ngc-learn.readthedocs.io/">documentation</a>).

For those contributing models/algorithms in either the `exhibitors/` or `exhibits/` directories, please send us an [email](mailto:ago@cs.rit.edu) if you are interested in writing your own walkthrough for us to include and integrate related to a particular model exhibit that you are working on in the official ngc-learn documentation as we warmly welcome the community to contribute to ngc-museum, as it is these contributions that help ensure various models of biomimetic inference/learning and brain-inspired computing see application as well as inspire future lines of scientific inquiry.

## Current Model Exhibits in the Museum
<b>Models with Spiking Dynamics</b>:<br>
1. Spiking neural network, trained with broadcast feedback alignment:
   <a href="https://github.com/NACLab/ngc-museum/tree/main/exhibits/bfa_snn">Model</a>,
   <a href="https://ngc-learn.readthedocs.io/en/latest/museum/snn_bfa.html">Walkthrough / Tutorial</a>
2. Diehl and Cook spiking network, trained with spike-timing-dependent plasticity (STDP):
   <a href="https://github.com/NACLab/ngc-museum/tree/main/exhibits/diehl_cook_snn">Model</a>,
   <a href="https://ngc-learn.readthedocs.io/en/latest/museum/snn_dc.html">Walkthrough / Tutorial</a>
3. Patch-level spiking network, trained with event-driven STDP:
   <a href="https://github.com/NACLab/ngc-museum/tree/main/exhibits/evstdp_patches">Model</a>
4. A self-supervised spiking neural circuit, trained via contrastive-signal dependent plasticity (CSDP):
   <a href="https://github.com/NACLab/ngc-museum/tree/main/exhibitors/nac_lab/csdp_snn">Model</a> <!--<a href="">Paper</a>-->
5. A spiking neural network for reinforcement learning, trained via MSTDP-ET:
   <a href="https://github.com/NACLab/ngc-museum/tree/main/exhibits/rl_snn">Model</a> <!--<a href="">Paper</a>-->
<!--
6. Patch-level spiking neural circuit trained with time-integrated STDP (TI-STDP)
   <a href="">Model</a>, 
   <a href="">Walkthrough / Tutorial</a>
-->

<b>Models with Graded Dynamics</b>:<br>
1. Discriminative predictive coding:
   <a href="https://github.com/NACLab/ngc-museum/tree/main/exhibits/pc_discrim">Model</a>,
   <a href="https://ngc-learn.readthedocs.io/en/latest/museum/pcn_discrim.html">Walkthrough / Tutorial</a>
2. Sparse coding (e.g., a Cauchy prior model and iterative sparse-thresholding / ISTA), trained with 2-factor Hebbian learning:
   <a href="https://github.com/NACLab/ngc-museum/tree/main/exhibits/olshausen_sc">Model</a>, 
   <a href="https://ngc-learn.readthedocs.io/en/latest/museum/sparse_coding.html">Walkthrough / Tutorial</a>
<!--
3. Deep fast iterative thresholding (FISTA), trained via 2-factor Hebbian learning: 
   <a href="">Model</a>, 
   <a href="">Walkthrough / Tutorial</a>
-->
3. Reconstructive hierarchical predictive coding:
   <a href="https://github.com/NACLab/ngc-museum/tree/main/exhibits/pc_recon">Model</a>
   <!--<a href="">Walkthrough</a>-->
4. Harmonium (restricted Boltzmann machine; RBM) with stochastic binary neurons, trained via contrastive divergence:
   <a href="https://github.com/NACLab/ngc-museum/tree/main/exhibits/harmonium">Model</a>, 
   <a href="">Walkthrough / Tutorial</a>
5. Sparse identification of nonlinear dynamics (SINDy): 
   <a href="https://github.com/NACLab/ngc-museum/tree/main/exhibits/sindy">Model</a>, 
   <a href="https://ngc-learn.readthedocs.io/en/latest/museum/sindy.html">Walkthrough / Tutorial</a> 
6. NeuralODE / Deep learning-driven model discovery (DeepMod): 
   <a href="https://github.com/NACLab/ngc-museum/tree/main/exhibits/DeepMoD_PC">Model</a>
<!--
7. A forward-only neural predictor trained via signal propagation:
   <a href="">Model</a>, 
   <a href="">Walkthrough / Tutorial</a>
8. A neural classifier trained via forward-forward learning:
   <a href="">Model</a>, 
   <a href="">Walkthrough / Tutorial</a>
-->

This package is distributed under the 3-Clause BSD license.<br>
It is currently maintained by the <a href="https://www.cs.rit.edu/~ago/nac_lab.html">Neural Adaptive Computing (NAC) laboratory</a>.
