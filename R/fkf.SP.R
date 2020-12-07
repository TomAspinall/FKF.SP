#'Fast Kalman Filtering using Sequential Processing.
#'
#'@description
#'\loadmathjax
#'The \code{fkf.SP} function performs fast and flexible Kalman Filtering using Sequential Processing. It is designed for efficient parameter
#'estimation through maximum likelihood estimation. \code{fkf.SP} wraps the C-function \code{fkf_SP} which relies upon the linear algebra subroutines of BLAS.
#'Sequential Processing is a univariate treatment of a multivariate series of observations and can benefit from computational efficiencies over
#'traditional Kalman Filtering. \code{fkf.SP} is based from the \code{fkf} function of the \code{FKF} package but, in general, is a faster Kalman Filtering method.
#'fkf.SP is compatible with missing observations.
#'
#'@param a0 A \code{vector} giving the initial value/estimation of the state variable
#'@param P0 A \code{matrix} giving the variance of a0
#'@param dt A \code{matrix} giving the intercept of the transition equation
#'@param ct A \code{matrix} giving the intercept of the measaurement equation
#'@param Tt An \code{array} giving factor of the transition equation
#'@param Zt An \code{array} giving the factor of the measurement equation
#'@param HHt An \code{array} giving the variance of the innovations of the transition equation
#'@param GGt A \code{vector} giving the diagonal elements of the \code{matrix} for the variance of the disturbances of the measurement equation. Covariance between disturbances
#'is not supported under the Sequential Processing method.
#'@param yt A \code{matrix} containing the observations. "NA"- values are allowed
#'
#'@details
#'
#'
#'\bold{Parameters}:
#'
#'The \code{fkf.SP} function builds upon the \code{fkf} function of the \code{FKF} package by adjusting the Kalman Filtering algorithm to
#'utilize sequential processing. The \code{fkf.SP} and \code{fkf} functions feature highly similar
#'arguments for compatibility purposes; only argument \code{GGt} has changed from an \code{array} type object to a \code{vector} or \code{matrix} type object.
#'The \code{fkf.SP} function takes the additional assumption over the \code{fkf} function that the variance of the disturbances of the measurement
#'equation are independent; a requirement of sequential processing (see below).
#'
#'The parameters can either be constant or deterministic
#'time-varying. Assume the number of discrete time observations is \mjeqn{n}{n}
#'i.e. \mjeqn{y = y_t}{y = y_t} where \mjeqn{t = 1, \cdots, n}{t = 1, \cdots, n}. Let \mjeqn{m}{m} be the
#'dimension of the state variable and \mjeqn{d}{ASCII representation} the dimension of the observations. Then, the parameters admit the following
#'classes and dimensions:
#'
#' \tabular{ll}{
#'     \code{dt} \tab either a \mjeqn{m \times n}{m * n} (time-varying) or a \mjeqn{m \times 1}{m * 1} (constant) matrix. \cr
#'     \code{Tt} \tab either a \mjeqn{m \times m \times n}{m * m * n} or a \mjeqn{m \times m \times 1}{m * m * 1} array. \cr
#'     \code{HHt} \tab either a \mjeqn{m \times m \times n}{m * m * n} or a \mjeqn{m \times m \times 1}{m * m * 1} array. \cr
#'     \code{ct} \tab either a \mjeqn{d \times n}{d * n} or a \mjeqn{d \times 1}{d * 1} matrix. \cr
#'     \code{Zt} \tab either a \mjeqn{d \times m \times n}{d * m * n} or a \mjeqn{d \times m \times 1}{d * m * 1} array. \cr
#'     \code{GGt} \tab either a \mjeqn{d \times n}{d * n} (time-varying) or a \mjeqn{d \times 1}{d * 1} matrix. \cr
#'     \code{yt} \tab a \mjeqn{d \times n}{d * n} matrix.
#'   }
#'
#'\bold{State Space Form}
#'
#'The following notation follows that of Koopman \emph{et al.} (1999) and the documentation of the \code{fkf} function. The Kalman Filter is characterised by the transition and measurement equations:
#'
#'\mjdeqn{\alpha_{t + 1} = d_t + T_t \cdot \alpha_t + H_t \cdot \eta_t}{alpha(t + 1) = d(t) + T(t) alpha(t) + H(t) * eta(t)}
#'\mjdeqn{y_t = c_t + Z_t \cdot \alpha_t + G_t \cdot \epsilon_t}{y(t) = c(t) + Z(t) alpha(t) + G(t) * epsilon(t)}
#'
#'
#'where \mjeqn{\eta_t}{eta(t)} and \mjeqn{\epsilon_t}{epsilon(t)} are iid
#' \mjeqn{N(0, I_m)}{N(0, I(m))} and iid \mjeqn{N(0, I_d)}{N(0, I(d))},
#' respectively, and \mjeqn{\alpha_t}{alpha(t)} denotes the state
#' vector. The parameters admit the following dimensions:
#'
#' \tabular{lll}{
#' \mjeqn{a_t \in R^m}{a[t] \in R^m} \tab \mjeqn{d_t \in R^m}{d[t] \in R^m} \tab \mjeqn{\eta_t \in R^m}{eta[t] \in R^m} \cr
#' \mjeqn{T_t \in R^{m \times m}}{T[t] \in R^(m * m )} \tab \mjeqn{H_t \in R^{m \times m}}{H[t]
#' \in R^(m * m)} \tab \cr
#' \mjeqn{y_t \in R^d}{y[t] in R^d} \tab \mjeqn{c_t \in R^d}{c[t] \in R^d} \tab \mjeqn{\epsilon_t \in R^d}{epsilon[t] \in R^d} \cr
#' \mjeqn{Z_t \in R^{d \times m}}{Z[t] \in R^(d * m)} \tab \mjeqn{G_t \in R^{d \times d}}{G[t]
#' \in R^(d * d)} \tab
#' }
#'
#' Note that \code{fkf.SP} takes as input \code{HHt} and \code{GGt} which corresponds to \mjeqn{H_t H_t'}{H[t] * t(H[t])} and
#' \mjeqn{diag(G_t)^2}{diag(G[t])^2} respectively.
#'
#'
#'\bold{Sequential Processing Iteration}:
#'
#'Traditional Kalman Filtering takes the entire observational vector \mjeqn{y_t}{y[t]} as the items for analysis. Sequential Processing
#'is an alternate approach that filters the elements of \mjeqn{y_t}{y[t]} one at a time. Let \mjeqn{p}{p} equal the number of observations
#'at time \mjeqn{t}{t} (ie. when considering possible missing observations \mjeqn{p \leq {d}}{p <= d}) . This univariate treatment of the multivariate series
#'has the advantage that the function of the covariance matrix, \mjeqn{F_t}{F[t]}, becomes \mjeqn{1 \times 1}{1 * 1}, avoiding the inversion of
#'a \mjeqn{p \times p}{p * p} matrix. This can provide computational efficiencies (especially under the case of many observations, ie. \mjeqn{p}{p} is large)
#'
#'The SP iteration involves treating the vector series: \mjeqn{y_1,\cdots,y_n}{y_1,\cdots,y_n} instead as the scalar series
#'\mjeqn{y_{1,1},\cdots,y_{(1,p)},y_{2,1},\cdots,y_{(n,p_n)}}{y_{1,1},\cdots,y_{(1,p)},y_{2,1},\cdots,y_{(n,p_n )}}
#'
#'For any time point, the observation vector is given by:
#'
#'\mjdeqn{y_t'=(y_{(t,1)},\cdots,y_{(t,p)} )}{y_t'=(y_(t,1),\cdots,y_(t,p) )}
#'
#'The filtering equations are written as:
#'
#'\mjdeqn{a_{t,i+1} = a_{t,i} + K_{t,i} v_{t,i}}{A}
#'\mjdeqn{P_{t,i+1} = P_{t,i} - K_{t,i} F_{t,i} K_{t,i}'}{A}
#'Where:
#'\mjdeqn{\hat y_{t,i} = c_t + Z_t \cdot a_{t,i}}{^y_t = c[t] + Z[t] * a_[t,i]}
#'\mjdeqn{v_{t,i}=y_{t,i}-\hat y_{t,i}}{A}
#'\mjdeqn{F_{t,i} = Z_{t,i} P_{t,i} Z_{t,i}'+ GGt_{t,i}}{GGt[t,i]}
#'\mjdeqn{K_{t,i} = P_{t,i} Z_{t,i}' F_{t,i}^{-1}}{A}
#'\mjdeqn{i = 1, \cdots, p}{i = 1, ..., p}
#'
#'Transition from time \mjeqn{t}{t} to \mjeqn{t+1}{t+1} occurs through the standard transition equations.
#'
#'\mjdeqn{\alpha_{t + 1,1} = d_t + T_t \cdot \alpha_{t,p}}{alpha(t + 1,1) = d(t) + T(t) alpha(t,p)}
#'
#'\mjdeqn{P_{t + 1,1} = T_t \cdot P_{t,p} \cdot T_t' + HHt}{P(t + 1,1) = T(t) P(t,p) T(t)' + H(t)}
#'
#'The log-likelihood at time \mjeqn{t}{t} is given by:
#'
#'\mjdeqn{log L_t = -\frac{p}{2}log(2\pi) - \frac{1}{2}\sum_{i=1}^p(log F_i + \frac{v_i^2}{F_i})}{log L_t = -p/2 * log(2 * pi) - 1/2 * sum_{i=1}^p (log F_i + (v_i^2)/F_i)}
#'
#'Where the log-likelihood of observations is:
#'
#'\mjdeqn{log L = \sum_t^n{log L_t}}{log L = \sum_t^n{log L_t}}
#'
#'
#'There's several distinctions between the Multivariate Kalman Filter and the Sequential processing filtered values.
#'The elements of the innovation vector \mjeqn{v_t}{v_t} are not the same as \mjeqn{v_{t,i}}{v[t,i]} for \mjeqn{i=1,\cdots,p}{i=1,\cdots,p}.
#'This is also true of the diagonal elements of the variance matrix \mjeqn{F_t}{F_t} and the variances \mjeqn{F_{t,i}}{F_{t,i}}, for \mjeqn{i=1,\cdots,p}{i=1,\cdots,p};
#'only the first diagonal element of \mjeqn{F_t}{F_t} is equal to \mjeqn{F_{t,1}}{F[t,1]}.
#'
#'@return
#'A \code{numeric} value corresponding to the log-likelihood calculated by the Kalman Filter. Ideal for maximum likelihood estimation through optimization routines such as \code{optim}.
#'
#'\bold{fkf and fkf.SP values}:
#'
#'Outputs of the \code{fkf} and \code{fkf.SP} functions differ slightly in that \code{fkf} returns a list object of filtered values returned by the
#'algorithm, whereas \code{fkf.SP} returns only a \code{numeric} value corresponding to the log-likelihood returned by the filter. \code{fkf}
#'is thus appropriate when filtered values are required and \code{fkf.SP} is appropriate for parameter estimation through maximum likelihood
#'methods.
#'
#'When there are no missing observations (ie. "NA" values) in \code{matrix} yt, the return of function \code{fkf.SP} and the \code{logLik}
#'object returned within the list of function \code{fkf} are identical. When there are NA values, however, the log-likelihood score returned
#'by \code{fkf.SP} is always higher. The log-likelihood score of the \code{FKF} C code is instantiated through the calculation of
#'\mjeqn{- n \times d \times log(2\pi)}{ASCII representation}, where \mjeqn{n}{ASCII representation} is the number of columns of \code{matrix}
#'\code{yt} and \mjeqn{d}{ASCII representation} is the number of rows of \code{matrix} \code{yt}. Under the assumption that there are
#'missing observations, \mjeqn{d}{ASCII representation} would instead become \mjeqn{d_t}{ASCII representation}, where \mjeqn{d_t \leq d \forall t}{ASCII representation}.
#'Whilst this doesn't influence parameter estimation, because observation matrix \code{yt} and thus the offset resulting from this is kept constant during maximum likelihood estimation,
#'this does result in low bias of the log-likelihood scores output by the \code{fkf} function.
#'
#'@references
#'
#'Anderson, B. D. O. and Moore. J. B. (1979). \emph{Optimal Filtering} Englewood Cliffs: Prentice-Hall.
#'
#'Fahrmeir, L. and tutz, G. (1994) \emph{Multivariate Statistical Modelling Based on Generalized Linear Models.} Berlin: Springer.
#'
#'Koopman, S. J., Shephard, N., Doornik, J. A. (1999). Statistical algorithms for models in state space using SsfPack 2.2. \emph{Econometrics Journal}, Royal Economic Society, vol. 2(1), pages 107-160.
#'
#'Durbin, James, and Siem Jan Koopman (2012). \emph{Time series analysis by state space methods.} Oxford university press.
#'
#'David Luethi, Philipp Erb and Simon Otziger (2018). FKF: Fast Kalman Filter. R package version 0.1.5.
#'https://CRAN.R-project.org/package=FKF
#'
#'@examples
#'
#'# FKF.SP is suitable for parameter estimation, and thus the following examples
#'# showcase how to estimate parameters of different models.
#'
#'## <-------------------------------------------------------------------------------
#'##Example 1 - ARMA(2,1) model estimation.
#'## <-------------------------------------------------------------------------------
#'
#'n <- 1000
#'
#'## Set the AR parameters
#'
#'ar1 <- 0.6
#'ar2 <- 0.2
#'ma1 <- -0.2
#'sigma <- sqrt(0.2)
#'
#'## Sample from an ARMA(2, 1) process
#'a <- stats::arima.sim(model = list(ar = c(ar1, ar2), ma = ma1), n = n,
#'             innov = rnorm(n) * sigma)
#'
#'## Create a state space representation out of the four ARMA parameters
#'arma21ss <- function(ar1, ar2, ma1, sigma) {
#' Tt <- matrix(c(ar1, ar2, 1, 0), ncol = 2)
#' Zt <- matrix(c(1, 0), ncol = 2)
#' ct <- matrix(0)
#' dt <- matrix(0, nrow = 2)
#' GGt <- matrix(0)
#' H <- matrix(c(1, ma1), nrow = 2) * sigma
#' HHt <- H %*% t(H)
#' a0 <- c(0, 0)
#' P0 <- matrix(1e6, nrow = 2, ncol = 2)
#' return(list(a0 = a0, P0 = P0, ct = ct, dt = dt, Zt = Zt, Tt = Tt, GGt = GGt,
#'             HHt = HHt))
#'             }
#'
#'## The objective function passed to 'optim'
#'objective <- function(theta, yt, SP = F) {
#'sp <- arma21ss(theta["ar1"], theta["ar2"], theta["ma1"], theta["sigma"])
#'  ans <- fkf.SP(a0 = sp$a0, P0 = sp$P0, dt = sp$dt, ct = sp$ct, Tt = sp$Tt,
#'                Zt = sp$Zt, HHt = sp$HHt, GGt = sp$GGt, yt = yt)
#'  return(-ans)
#'}
#'
#'#Estimate Parameters using the optim function:
#'theta <- c(ar = c(0, 0), ma1 = 0, sigma = 1)
#'ARMA_MLE <- optim(theta, objective, yt = rbind(a), hessian = TRUE, SP = T)
#'
#'
#'## <-------------------------------------------------------------------------------
#'#Example 2 - Nile Example:
#'## <-------------------------------------------------------------------------------
#'#Local level model for the Nile's annual flow:
#'
#'## Transition equation:
#'## alpha[t+1] = alpha[t] + eta[t], eta[t] ~ N(0, HHt)
#'## Measurement equation:
#'## y[t] = alpha[t] + eps[t], eps[t] ~  N(0, GGt)
#'
#'yt <- Nile
#'
#'##Incomplete Nile Data - two NA's are present:
#'yt[c(3, 10)] <- NA
#'
#'## Set constant parameters:
#'dt <- ct <- matrix(0)
#'Zt <- Tt <- matrix(1)
#'a0 <- yt[1]   # Estimation of the first year flow
#'P0 <- matrix(100)       # Variance of 'a0'
#'
#'## Parameter estimation - maximum likelihood estimation:
#'##Unknown parameters initial estimates:
#'GGt <- HHt <- var(yt, na.rm = TRUE) * .5
#'set.seed(1)
#'#Perform Maximum Likelihood Estimation
#'Nile_MLE <- suppressWarnings(optim(c(HHt = HHt, GGt = GGt),
#'                                   fn = function(par, ...)
#'                                     -fkf.SP(HHt = matrix(par[1]), GGt = matrix(par[2]), ...),
#'                                   yt = rbind(yt), a0 = a0, P0 = P0, dt = dt, ct = ct,
#'                                   Zt = Zt, Tt = Tt))
#'## <-------------------------------------------------------------------------------
#'#Example 3 - Dimensionless Treering Example:
#'## <-------------------------------------------------------------------------------
#'
#'## Transition equation:
#'## alpha[t+1] = alpha[t] + eta[t], eta[t] ~ N(0, HHt)
#'## Measurement equation:
#'## y[t] = alpha[t] + eps[t], eps[t] ~  N(0, GGt)
#'
#'## tree-ring widths in dimensionless units
#'y <- treering
#'
#'## Set constant parameters:
#'dt <- ct <- matrix(0)
#'Zt <- Tt <- matrix(1)
#'a0 <- y[1]            # Estimation of the first width
#'P0 <- matrix(100)     # Variance of 'a0'
#'
#'##Time comparison - Estimate parameters 10 times:
#'Treering_MLE <- suppressWarnings(optim(c(HHt = var(y, na.rm = TRUE) * .5,
#'                        GGt = var(y, na.rm = TRUE) * .5),
#'                      fn = function(par, ...)
#'                        -fkf.SP(HHt = array(par[1],c(1,1,1)), GGt = matrix(par[2]), ...),
#'                      yt = rbind(y), a0 = a0, P0 = P0, dt = dt, ct = ct,
#'                      Zt = Zt, Tt = Tt))
#'
#'##Not run - Filter tree ring data with estimated parameters using the FKF package:
#'#fkf.obj <- fkf(a0, P0, dt, ct, Tt, Zt, HHt = array(fit.fkf$par[1],c(1,1,1)),
#'#               GGt = matrix(fit.fkf$par[2]), yt = rbind(y))
#'@export
fkf.SP = function (a0, P0, dt, ct, Tt, Zt, HHt, GGt, yt){
  inputs <- c("a0", "P0", "dt", "ct", "Tt", "Zt", "HHt", "GGt", "yt")

  missing.values <- c(missing(a0), missing(P0), missing(dt), missing(ct),
                     missing(Tt), missing(Zt), missing(HHt), missing(GGt),
                     missing(yt))
  if(any(missing.values)) stop(paste("Input Arguments", paste(inputs[missing.values], collapse = ", "), "missing."))

  ##Class type check:
  stored_inputs <- c(a0 = storage.mode(a0), P0 = storage.mode(P0),
                 dt = storage.mode(dt), ct = storage.mode(ct), Tt = storage.mode(Tt),
                 Zt = storage.mode(Zt), HHt = storage.mode(HHt), GGt = storage.mode(GGt),
                 yt = storage.mode(yt))
  invalid_stored_inputs = any(stored_inputs != "double")
  if(invalid_stored_inputs) stop(paste("Input Arguments", paste(inputs[invalid_stored_inputs], collapse = ", "), "not of type 'double'"))

  ##Invalid Input Classes:
  error.string <- ""
  if (!is.vector(a0)) error.string <- paste0(error.string, "'a0' must be of class 'vector'!\n")
  if (!is.matrix(P0)) error.string <- paste0(error.string, "'P0' must be of class 'matrix'!\n")
  if (!is.matrix(dt) && !is.vector(dt)) error.string <- paste0(error.string, "'dt' must be of class 'vector' or 'matrix'!\n")
  if (!is.matrix(ct) && !is.vector(ct)) error.string <- paste0(error.string, "'ct' must be of class 'vector' or 'matrix'!\n")
  if (!is.array(Tt)) error.string <- paste0(error.string, "'Tt' must be of class 'matrix' or 'array'!\n")
  if (!is.array(Zt)) error.string <- paste0(error.string, "'Zt' must be of class 'matrix' or 'array'!\n")
  if (!is.array(HHt)) error.string <- paste0(error.string, "'HHt' must be of class 'matrix' or 'array'!\n")
  if (!is.matrix(GGt) && !is.vector(GGt)) error.string <- paste0(error.string, "'GGt' must be of class 'vector' or 'matrix'!\n")
  if (error.string != "") stop(error.string)

  ##Inputs argument class consistency:
  if(is.vector(dt)) dt <- as.matrix(dt)
  if(is.vector(ct)) ct <- as.matrix(ct)
  if(is.matrix(Tt)) Tt <- array(Tt, c(dim(Tt), 1))
  if(is.matrix(Zt)) Zt <- array(Zt, c(dim(Zt), 1))
  if(is.vector(GGt)) GGt <- as.matrix(GGt)
  if(is.matrix(HHt)) HHt <- array(HHt, c(dim(HHt), 1))

  ###Checking for invalid dimensions in inputs:

  #m:
  m <- length(a0)
  invalid_m_dimensions <- c(dim(P0), dim(dt)[1], dim(Zt)[2], dim(HHt)[1:2], dim(Tt)[1:2]) != m
  m_dimension_variable <- c("P0", "dt", "Zt", rep("HHt",2), rep("Tt",2))
  if(any(invalid_m_dimensions)) stop(paste("Dimensions in", paste(m_dimension_variable[invalid_m_dimensions], collapse = ", "), "do not match length of state vector ('m')"))

  #n:
  n <- dim(yt)[2]
  n_dimension_check = c(dim(dt)[2], dim(ct)[2], dim(Tt)[3], dim(Zt)[3], dim(HHt)[3], dim(GGt)[2])
  n_dimensions = c("dt", "ct", "Tt", "Zt", "HHt")
  invalid_n_dimensions = (n_dimension_check != n) & (n_dimension_check != 1)
  if(any(invalid_n_dimensions)) stop(paste("Dimensions in", paste(n_dimensions[invalid_n_dimensions], collapse = ", "), "do not equal either 1 or ncol(yt) ('n')"))

  #d:
  d <- nrow(yt)
  invalid_d_dimensions = c(dim(ct)[1], dim(Zt)[1], dim(GGt)[1]) != d
  d_dimensions = c("ct", "Zt", "GGt")
  if(any(invalid_d_dimensions)) stop(paste("Dimensions in", paste(d_dimensions[invalid_d_dimensions], collapse = ", "), "do not equal nrow(yt) ('d')"))


  ###Call Kalman Filter Function:
  ans <- .Call("fkf_SP", a0, P0, dt, ct, Tt, Zt, HHt, GGt,
               yt, PACKAGE = "FKF.SP")

  if(is.na(ans))  warning(paste("Log-Likelihood returned NA"))
  return(ans)
}

