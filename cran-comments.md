## Test environments
- local R installation, R 4.2.1
* win-builder (devel)
* R-hub Windows Server 2022 64-bit (r-devel)
* R-hub ubuntu-gcc-release (r-release)
* R-hub fedora-clang-gfortran-devel (r-devel)

## R CMD check results

0 errors | 0 warnings | 1 note

Namespace in Imports field not imported from: 'mathjaxr'
    All declared Imports should be used.

'mathjaxr' is used to include equations in the Rd files. It is called using the \loadmathjax macro, and hence is used in the package despite the R CMD check note.

* This release addresses the problems as requested by the CRAN team.

## Downstream Dependencies

'NFCP' is a downstream dependency. I have run R CMD check, which passed with 0 errors, warnings and notes.

* This is an update of an existing release (0.3.0 -> 0.3.1)
