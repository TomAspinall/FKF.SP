#' Fast Kalman Smoother Through Sequential Processing:
#'
#' @description This function can be run after running \code{\link{fkf.SP}} to produce
#' "smoothed" estimates of the state variable \eqn{\alpha_t}{alpha(t)}.
#' Unlike the output of the filter, these estimates are conditional
#' on the entire set of \eqn{n}{n} data points rather than only the past, see details.
#'
#' @param FKF.SP_obj  An S3-object of class "fkf.SP", returned by \code{\link{fkf.SP}}.
#'
#' @return An S3-object of class "fks.SP" which is a list with the following elements:
#'
#'   \code{ahatt}  A \eqn{m \times n}{m * n}-matrix containing the
#'   smoothed state variables, i.e. ahatt[,t] = \eqn{a_{t|n}}{a(t|n)}\cr
#'   \code{Vt}  A \eqn{m \times m \times n}{m * m * n}-array
#'   containing the variances of \code{ahatt}, i.e. Vt[,,t] = \eqn{P_{t|n}}{P(t|n)}\cr
#'
#' @details
#' temporary
#'
#' @section References:
#'
#' Koopman, S. J. and Durbin, J. (2000). \emph{Fast filtering and smoothing for multivariate state space models} Journal of Time Series Analysis Vol. 21, No. 3
#' 
#'@examples
#'### Perform Kalman Filtering and Smoothing through sequential processing:
#'#Nile's annual flow:
#'yt <- Nile
#'
#'# Incomplete Nile Data - two NA's are present:
#'yt[c(3, 10)] <- NA
#'
## Set constant parameters:
#'dt <- ct <- matrix(0)
#'Zt <- Tt <- matrix(1)
#'a0 <- yt[1]   # Estimation of the first year flow
#'P0 <- matrix(100)       # Variance of 'a0'
#'
#'# Parameter estimation - maximum likelihood estimation:
#'# Unknown parameters initial estimates:
#'GGt <- HHt <- var(yt, na.rm = TRUE) * .5
#'HHt = matrix(HHt)
#'GGt = matrix(GGt)
#'yt = rbind(yt)
#'# Filter through the Kalman filter - sequential processing:
#'Nile_filtered <- fkf.SP(HHt = matrix(HHt), GGt = matrix(GGt), a0 = a0, P0 = P0, dt = dt, ct = ct,
#'                   Zt = Zt, Tt = Tt, yt = rbind(yt), verbose = TRUE)
#'# Smooth filtered values through the Kalman smoother - sequential processing:
#'Smoothed_Estimates <- fks_SP(Nile_filtered)
#'
#' @export
fks_SP <- function (FKF.SP_obj) {
  if (class(FKF.SP_obj) != 'fkf.SP') stop('Input must be an object of class FKF.SP')

  yt <- FKF.SP_obj$yt
  vt <- FKF.SP_obj$vt
  Ftinv <- FKF.SP_obj$Ftinv
  n <- dim(yt)[2]
  Kt <- FKF.SP_obj$Kt
  att <- FKF.SP_obj$att[,1:n]
  Ptt <- FKF.SP_obj$Ptt[,,1:n]
  Zt <- FKF.SP_obj$Zt
  Tt <- FKF.SP_obj$Tt
  
  return(.Call("fks_SP", Tt, Zt, yt, vt, Kt, Ftinv, att, Ptt, PACKAGE = "FKF.SP"))
}