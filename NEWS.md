# FKF.SP 0.3.0

More outputs are now returned from the 'fkf.SP' function when 'verbose' is set to true. These are 'yt', 'Tt', 'Zt', 'Ftinv', 'vt', 'Kt'

The 'fks.SP' function has been added, which is a Kalman smoothing implementation written in compiled C code and using the solution described in the textbook of Durbin and Koopman (2001): "Time Series Analysis by State Space Methods).

# FKF.SP 0.2.0

The 'fkf.SP' function now includes the 'verbose' argument, allowing for filtered values to be returned as a list object.

The default setting for the 'verbose' argument is FALSE, resulting in no changes to existing applications of this function.

# FKF.SP 0.1.3

- USE_FC_LEN_T now adopted, consistent with its upcoming obligatory usage

# FKF.SP 0.1.2

- Minor name changes to the vignette
- Deprecated data from the 'NFCP' package used in vignette has been updated

# FKF.SP 0.1.1

- Minor fixes in error reporting of 'fkf.SP'.
- Example 4 of the vignette added
- Now suggests the 'NFCP' package for example 4
- Minor documentation edits
- 'fkf.SP' no longer returns a 'warning' when NA's are returned


# FKF.SP 0.1.0

- Release of FKF.SP
