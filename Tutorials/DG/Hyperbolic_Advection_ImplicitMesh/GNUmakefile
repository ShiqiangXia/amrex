# PATH TO AMREX #######################################################
AMREX_HOME ?= ../../../
#######################################################################

COMP = gnu
DEBUG = TRUE
USE_MPI = FALSE	
USE_CUDA = FALSE

DIM = 2

include ./Make.package

include $(AMREX_HOME)/Tools/GNUMake/Make.defs

include $(AMREX_HOME)/Src/Base/Make.package
include $(AMREX_HOME)/Src/Boundary/Make.package

include $(AMREX_HOME)/Src/DG/Make.package
include $(AMREX_HOME)/Src/DG/Algoim/Make.package
include $(AMREX_HOME)/Src/DG/LinAlg/blas/Make.package
include $(AMREX_HOME)/Src/DG/LinAlg/lapack/Make.package

include $(AMREX_HOME)/Tools/GNUMake/Make.rules
