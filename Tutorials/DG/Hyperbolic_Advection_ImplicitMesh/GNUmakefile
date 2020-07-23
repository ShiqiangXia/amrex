# PATH TO AMREX #######################################################
AMREX_HOME ?= ../../../
#######################################################################



DEBUG = FALSE
DIM = 2

COMP = gcc
CXXSTD = c++14 -g

USE_MPI = FALSE

USE_CUDA = FALSE



include ./Make.package

include $(AMREX_HOME)/Tools/GNUMake/Make.defs

include $(AMREX_HOME)/Src/Base/Make.package
include $(AMREX_HOME)/Src/Boundary/Make.package

include $(AMREX_HOME)/Src/DG/Make.package
include $(AMREX_HOME)/Src/DG/Algoim/Make.package
include $(AMREX_HOME)/Src/DG/LinAlg/blas/Make.package
include $(AMREX_HOME)/Src/DG/LinAlg/lapack/Make.package

include $(AMREX_HOME)/Tools/GNUMake/Make.rules
