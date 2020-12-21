# TSCPM
Term Structure N-Factor Commodity Pricing Models

<!-- badges: start -->
<!-- badges: end -->

'TSCPM' (Term Structure Commodity Pricing Models) provides a framework for the modeling, parameter estimation, probabilistic forecasting, option valuation and simulation of commodity prices through state space and Monte Carlo methods, risk-neutral valuation and Kalman filtering. Commodity pricing models are (systems of) stochastic differential equations that are utilized for the valuation and hedging of commodity contingent claims (ie. derivative products on the commodity) and other commodity related investments. Commodity pricing models that capture market dynamics are of great importance to commodity market participants in order to exercise sound investment and risk-management strategies. The n-factor commodity pricing model framework was first presented in the work of Cortazar and Naranjo (2006) <doi: 10.1002/fut.20198>. Kalman filtering is performed using sequential processing through the 'FKF.SP' function to maximise computational efficiency. Parameter estimation is performed using genetic algorithms through the 'rGenoud' package to ensure a global maximum is reached during maximum likelihood estimation and optimal parameters are estimated. Examples presented throughout 'TSCPM' replicate the two-factor crude oil commodity pricing model presented in the prolific work of Schwartz and Smith (2000) <doi: 10.1287/mnsc.46.7.893.12034>.

The primary features of 'TSCPM' package include:

- Parameter estimation of commodity pricing models through state space methods, Kalman filtering and maximum likelihood estimation.

- Analytic valuation of European call and put option prices under estimated commodity pricing models

- Probabilistic forecasting and Monte Carlo simulation of future commodity price paths.

## Installation

You can install the released version of TSCPM from [CRAN](https://CRAN.R-project.org) with:

```
install.packages("TSCPM")
```

And the development version from [GitHub](https://github.com/) with:

```
devtools::install_github("TomAspinall/TSCPM")
```
which contains source code for the package starting with version 0.1.0.
