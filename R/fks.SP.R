#'@examples
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