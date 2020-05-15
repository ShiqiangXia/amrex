AMREX_HOME ?= ../../../

DEBUG	= FALSE
DEBUG	= TRUE

DIM	= 2

COMP    = gcc
CXXSTD = c++14

USE_MPI   = TRUE
USE_OMP   = FALSE
USE_CUDA  = FALSE
USE_HIP   = FALSE

include $(AMREX_HOME)/Tools/GNUMake/Make.defs

include ./Make.package
include $(AMREX_HOME)/Src/Base/Make.package

include $(AMREX_HOME)/Tools/GNUMake/Make.rules
