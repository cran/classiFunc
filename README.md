
<!-- README.md is generated from README.Rmd. Please edit that file -->

# classiFunc

[![Travis-CI Build
Status](https://api.travis-ci.org/maierhofert/classiFunc.svg?branch=master)](https://travis-ci.org/r-lib/classiFunc)
[![packageversion](https://img.shields.io/badge/Package%20version-0.1.1-orange.svg?style=flat-square)](commits/master)
[![CRAN\_Status\_Badge](https://www.r-pkg.org/badges/version/classiFunc)](https://cran.r-project.org/package=classiFunc)
[![](https://cranlogs.r-pkg.org/badges/classiFunc)](https://cran.r-project.org/package=classiFunc)
[![Licence](https://img.shields.io/badge/licence-GPL--3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0.en.html)

## Overview

The classiFunc package implements methods for functional data
classification. The main functions of this package are classiKnn, a k
nearest neighbor estimator for functional data, and classiKernel, a
kernel estimator for functional data. The package uses efficiently
implemented semimetrics to create the distance matrix of the functional
observations in the function computeDistMat.

## Using classiFunc

For installation instructions, see below. A hands on introduction to 
can be found in the
[vignette](https://cran.r-project.org/package=classiFunc/vignettes/classiFunc.html).
Details on specific functions are in the [reference
manual](https://cran.r-project.org/package=classiFunc/classiFunc.pdf).

## Issues & Feature Requests

For issues, bugs, feature requests etc. please use the [Github
Issues](https://github.com/maierhofert/classiFunc/issues). Input is
always welcome.

## Installation

You can install the current classiFunc version from CRAN with:

``` r
install.packages("classiFunc")
```

or the latest patched version from Github with:

``` r
# install.packages("devtools")
devtools::install_github("maierhofert/classiFunc")
```

<!-- ## Example -->

<!-- This is a basic example which shows you how to solve a common problem: -->

<!-- ```{r example} -->

<!-- ## basic example code -->

<!-- ``` -->
