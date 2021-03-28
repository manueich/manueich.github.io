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

## The Variational Bayes Approach

We start by introducing the PDF $q(w,\beta)$ which we will use an approximation to the true posterior distribution $p(w,\beta\|\mathcal{D})$. Next, we calculate the [Kullback-Leibler divergencee](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence), between the true and approximated distributions:

$$D_{KL}(q(w,\beta)||p(w,\beta|\mathcal{D}))=\int q(w,\beta)\log\frac{q(w,\beta)}{p(w,\beta|\mathcal{D})}dwd\beta$$

The Kullback-Leibler divergence has been introduced in the context of information theory. However, for its use in the VB approach, there are three properties we need to know: (1) it is always positive, (2) it provides a measure for the similarity between two distributions and (3) it is zero if they are identical. Substituting Bayes' rule for the posterior distribution we get:

$$\begin{aligned} D_{KL}(q(w,\beta)||p(w,\beta|\mathcal{D})) &=\int q(w,\beta)\log\frac{q(w,\beta)P(\mathcal{D})}{L(\mathcal{D}|w,\beta)p(w,\beta)}dwd\beta\\ &=\int q(w,\beta) \left[\log q(w,\beta) + \log P(\mathcal{D}) - \log L(\mathcal{D}|w,\beta) - \log p(w,\beta)\right]dwd\beta\\ &= \mathbb{E}\left[\log q(w,\beta) + \log P(\mathcal{D}) - \log L(\mathcal{D}|w,\beta) - \log p(w,\beta)\right]_{q(w,\beta)}\\ &= \mathbb{E}\left[\log q(w,\beta) - \log L(\mathcal{D}|w,\beta) - \log p(w,\beta)\right]_{q(w,\beta)} + \log P(\mathcal{D})\\ \end{aligned}$$

In the last step, we have pulled out the log marginal likelihood from the expectation as it is independent from $w$ and $\beta$. Introducing the Free Energy $\mathcal{F}$ and doing some rearranging we get:

$$\underbrace{\mathbb{E}\left[\log p(w,\beta)  + \log L(\mathcal{D}|w,\beta) - \log q(w,\beta)\right]_{q(w,\beta)}}_{\mathcal{F}} = \log P(\mathcal{D}) - D_{KL}(q(w,\beta)||p(w,\beta|\mathcal{D}))$$

Now we can see that the Free Energy can be considered a lower bound on the marginal likelihood, in the sense that if the posterior $p(w,\beta\|\mathcal{D})$ and approximated distributions $q(w,\beta)$ are identical, then $D_{KL}=0$, which in turn leads to $\mathcal{F}=\log P(\mathcal{D})$. We can also see that when maximising the Free Energy with respect to $q$, we are minimising $D_{KL}$. This in turn provides us with an approximation of the true posterior distribution through $q$ and simultaneously an lower bound on the model evidence through the Free Energy.

### Free Energy Maximisation

Maximising the Free Energy (which is a [functional](https://en.wikipedia.org/wiki/Functional_(mathematics))) with respect to $q$ is a problem that can be solved using the [Variational Calculus](https://en.wikipedia.org/wiki/Calculus_of_variations), thereby giving the whole method its name. In short, variational calculus gives the tools to optimise functionals (a function that takes another function as input). 

Returning to our problem at hand, we need to introduce the fist approximation of the distribution $q$ in order to maximise the Free Energy. Here, we say that the joint distribution of our unknown parameters can be factorised, i.e.

$$q(w,\beta) = q(w)q(\beta)$$

This is known as the mean-field approximation and assumes that there is no conditional dependency between the unknown parameters. The next step is to maximise the Free energy with the individual components of $q$. For that we set

$$\frac{\partial \mathcal{F}}{\partial q(w)}=0$$

$$\frac{\partial \mathcal{F}}{\partial q(\beta)}=0$$

and solve for $q(w)$ and $q{\beta}$, respectively. This solution is fairly complex and we advise the curious reader to this [link](https://bjlkeng.github.io/posts/variational-bayes-and-the-mean-field-approximation/), where the steps for the general case are explained. However, an understanding of these steps is not required to follow the rest of this tutorial, or indeed the VB approach, so we will just show the solution. The two densities $q^*(w)$ and $q^*(\beta)$ that maximise the Free energies can be calculated as follows:

$$\log q^*(w) = \mathbb{E}[\log L(\mathcal{D}|w,\beta)+\log p(w) + \log p(\beta)]_{q(\beta)}$$

$$\log q^*(\beta) = \mathbb{E}[\log L(\mathcal{D}|w,\beta)+\log p(w) + \log p(\beta)]_{q(w)}$$

Examining these two equations we can see that they are not independent, i.e. we need $q(\beta)$ to calculate $q^*(w)$ and vice versa. To get around this issue, the VB method employs an iterative maximisation approach, illustrated in the figure below. This means that we initialise both approximated distributions with their respective priors and iteratively update them, always keeping one component constant. After each iteration step, we recalculate the Free Energy. The process is terminated when the change in Free Energy from one step to the next is below some user defined threshold, typically $10^{-3}$ or lower. 

<center><img src="/_pages/Bayesian_Inference/Fmax.png" width="760" height="500"></center>

Figure 1: Iterative maximisation of the Free Energy.

### Update rules

In order to ass