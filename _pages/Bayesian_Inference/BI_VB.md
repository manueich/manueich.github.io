---
layout: archive
title: "Approximate Bayesian Inference using Variational Bayesian Analysis"
excerpt: "Tutorial on Approximate Bayesian Inference"
author_profile: false
---

In variational Bayesian (VB) analysis, the posterior distribution over unknown parameters is approximated by assuming it has certain properties such as factorising in a specific way or taking a specific parametric form. Alongside the approximation of the posterior, the VB method also provides an estimation of the marginal likelihood. 

## Recap

Before we can begin, we need establish some foundations with respect to the calculations of [expectations](https://en.wikipedia.org/wiki/Expected_value). The expectation of a function $f(x)$ with respect to some PDF $p(x)$ is expressed as follows:

$$\mathbb{E}[f(x)]_{p(x)}=\int f(x)p(x)dx$$

In words, the expectation gives us the expected value of $f(x)$ if we assume that the random variable $x$ follows the PDF $p(x)$. A simple example is the expectation of a Normal distribution:

$$\mathbb{E}[x]_{\mathcal{N}(x|\mu,\tau^{-1})}=\int x\mathcal{N}(x|\mu,\tau^{-1})dx=\mu$$

In words, we expect a random variable $x$ to take the value of the mean $\mu$ if it follows a normal distribution. As we have seen before, the expectation of a Gamma distribution is:

$$\mathbb{E}[x]_{\mathcal{Ga}(x|a,b)}=\int x\mathcal{Ga}(x|a,b)dx=\frac{a}{b}$$

A few more expectations that we will need to know are:

$$\mathbb{E}[x^2]_{\mathcal{N}(x|\mu,\tau^{-1})}=\mu^2+\frac{1}{\tau},$$

$$\mathbb{E}[\log x]_{\mathcal{Ga}(x|a,b)}=\psi(a)-\log b,$$

where $\psi(\cdot)$ is the [Digamma function](https://en.wikipedia.org/wiki/Digamma_function)

## The variational Bayes Approach

We start by introducing the PDF $q(w,\beta)$ which we will use an approximation to the true posterior distribution $p(w,\beta\|\mathcal{D})$. Next, we calculate the [Kullback-Leibler divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence), between the true and approximated distributions:

$$D_{KL}(q(w,\beta)||p(w,\beta\|\mathcal{D}))=\int q(w,\beta)\log\frac{q(w,\beta)}{p(w,\beta\|\mathcal{D})}dwd\beta$$

The Kullback-Leibler divergence has been introduced in the context of information theory. However, for its use in the VB approach, there are three properties we need to know: (1) it is always positive, (2) it provides a measure for the similarity between two distributions and (3) it is zero if they are identical. Substituting Bayes' rule for the posterior distribution we get:

$$\begin{aligned} D_{KL}(q(w,\beta)||p(w,\beta|\mathcal{D})) &=\int q(w,\beta)\log\frac{q(w,\beta)P(\mathcal{D})}{L(\mathcal{D}|w,\beta)p(w,\beta)}dwd\beta\\ &=\int q(w,\beta) \left[\log q(w,\beta) + \log P(\mathcal{D}) - \log L(\mathcal{D}|w,\beta) - \log p(w,\beta)\right]dwd\beta\\ &= \mathbb{E}\left[\log q(w,\beta) + \log P(\mathcal{D}) - \log L(\mathcal{D}|w,\beta) - \log p(w,\beta)\right]_{q(w,\beta)}\\\end{aligned}$$








... to be continued
