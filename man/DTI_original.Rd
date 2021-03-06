% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/data.R
\docType{data}
\name{DTI_original}
\alias{DTI_original}
\title{Diffusion Tensor Imaging: tract profiles and outcomes}
\format{A data frame made up of \describe{
\item{cca}{A 382 x 93
matrix of fractional anisotropy tract profiles from the corpus
callosum containing missing values;}
\item{rcst}{A 382 x 55 matrix
of fractional anisotropy tract profiles from the right corticospinal
tract containing missing values;}
\item{ID}{Numeric vector of subject ID numbers;}
\item{visit}{Numeric vector of the subject-specific visit
numbers;}
\item{visit.time}{Numeric vector of the subject-specific visit time, measured
in days since first visit;}
\item{Nscans}{Numeric vector indicating the total number of visits
for each subject;}
\item{case}{Numeric vector of multiple sclerosis case status: 0 - healthy control, 1 - MS case;}
\item{sex}{factor variable indicated subject's sex;}

\item{pasat}{Numeric vector containing the PASAT score at
each visit.}
}}
\description{
Fractional anisotropy (FA) tract profiles for the corpus callosum (cca) and
the right corticospinal tract (rcst). Accompanying the tract profiles are
the subject ID numbers, visit number, total number of scans, multiple
sclerosis case status and Paced Auditory Serial Addition Test (pasat)
score.
}
\details{
If you use this data as an example in written work, please include the
following acknowledgment: ``The MRI/DTI data were collected at Johns
Hopkins University and the Kennedy-Krieger Institute"

Data and description was copied from the \code{\link[refund:refund-package]{refund}} package.
}
\references{
Goldsmith, J., Bobb, J., Crainiceanu, C., Caffo, B., and Reich,
D. (2011). Penalized Functional Regression. \emph{Journal of Computational
and Graphical Statistics}, 20, 830 - 851.

Goldsmith, J., Crainiceanu, C., Caffo, B., and Reich, D. (2010).
Longitudinal Penalized Functional Regression for Cognitive Outcomes on
Neuronal Tract Measurements. \emph{Journal of the Royal Statistical
Society: Series C}, 61, 453 - 469.
}
