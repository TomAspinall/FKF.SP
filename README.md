
# FKF.SP

<!-- badges: start -->
<!-- badges: end -->

The R package FKF.SP provides a fast and flexible Kalman Filtering implementation utilizing Sequential Processing, designed for efficient parameter estimation through maximum likelihood estimation. Sequential Processing is a univariate treatment of a multivariate series of observations and can benefit from computational efficiencies over traditional Kalman Filtering. Sequential Processing takes the additional assumption that the white noise of observations are independent at each discrete time point. FKF.SP was built upon the existing FKF package.

## Installation

You can install the released version of FKF.SP from [CRAN](https://CRAN.R-project.org) with:

```
install.packages("FKF.SP")
```

And the development version from [GitHub](https://github.com/) with:

```
devtools::install_github("TomAspinall/FKF.SP")
```
which contains source code for the package starting with version 0.1.0.
