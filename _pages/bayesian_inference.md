---
layout: archive
title: "Tutorial on Bayesian Inference"
permalink: /bayesian_inference/
author_profile: true
---

## Aim and Scope

This tutorial exemplifies the use of approximate Bayesian inference with a variational Bayesian (VB) and Markov Chain Monte Carlo (MCMC) approach on a very simple linear regression model. It is aimed at people who already have a basic understanding of probability theory including Bayes' rule, probability density functions and expectations and want to understand the inner workings, advantages and disadvantages of VB and MCMC approaches. This is important to apply both approaches to more complex datasets and models.

## Introduction

Bayesian inference is a powerful tool to identify a variety of statistical models from which we can make make predictions and quantify the uncertainties we have in those predictions. This uncertainty is expressed in terms of probabilities, in particular through the use of probability density functions (PDF). Throughout this tutorial we will only deal with continuous random variables whose PDF is denoted by $p$, having the property $\int p(x) dx = 1$.

### The Basic Model

In this tutorial, we will discuss the following basic linear regression model:

$$y = w f(x) + \varepsilon$$
$$\varepsilon \sim \mathcal{N}(0,\beta^{-1})$$

The input and output (or target), collectively referred to as data $\mathcal{D}$, of the model are $x$ and $y$, respectively. The function $f$ can be any continous function of the input $x$. The model has two unkonwn parameters that we want to infer from the data $\mathcal{D}$: the slope parameter $w$ and the precision $\beta$ (inverse of the variance) of the measurement noise $\varepsilon$. For the purpose of illustration we will assume the parameter $w$ to be one dimensional. 

The goal of Bayesian inference in the context of this model is to infer the posterior distribution $p(w,\beta|\mathcal{D})$ of the unknown parameters given some data. Using Bayes' rule this posterior can be calculated as follows:

$$p(w,\beta|\mathcal{D})=\frac{L(\mathcal{D}|w,\beta)p(w,\beta)}{P(\mathcal{D})}$$

gfgfgf


\[[Link](Bayesian_Inference/BI_True.md)\]


