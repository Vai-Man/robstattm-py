# R-side validation for RobStatTM::locScaleM and RobStatTM::scaleM

suppressPackageStartupMessages(library(RobStatTM))

x1 <- c(1, 2, 2, 3, 100)
cat("locScaleM(x1):\n")
print(locScaleM(x1))
cat("scaleM(x1):\n")
print(scaleM(x1))

set.seed(123)
r <- rnorm(150, sd = 1.5)
cat("\nlocScaleM(r):\n")
print(locScaleM(r))
cat("scaleM(r):\n")
print(scaleM(r))

set.seed(123)
r2 <- c(rnorm(135, sd = 1.5), rnorm(15, mean = -10, sd = 0.5))
cat("\nlocScaleM(r2):\n")
print(locScaleM(r2))

set.seed(123)
r3 <- c(rnorm(135, sd = 1.5), rnorm(15, mean = -5, sd = 0.5))
cat("scaleM(r3, family='opt'):\n")
print(scaleM(r3, family = 'opt'))
