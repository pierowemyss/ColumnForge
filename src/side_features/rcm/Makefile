# Builds lib/RCM_solver.so -- the residue-curve integrator behind core/rcm.py.
#
#   make -C src/side_features/rcm
#
# RCM_solv.c carries no thermodynamics: it links against ../../native/nifco2.f90
# and calls its psat/gamma/phi routines as plain Fortran externals. Needs GSL
# (multiroot) and gfortran; minpack supplies hybrd1 to nifco2.

CC = gcc
# gfortran must match the arch of the Python that dlopens the result: on Apple
# silicon that is the /opt/homebrew gcc, not the x86_64 one at /usr/local.
# `?=` is no good here: make ships a built-in default for FC (f77), so it
# already counts as defined. Only override when nobody chose one.
ifeq ($(origin FC), default)
    FC := $(shell ls /opt/homebrew/bin/gfortran-* 2>/dev/null | tail -1)
    ifeq ($(FC),)
        FC := gfortran
    endif
endif

UNAME := $(shell uname)
ifeq ($(UNAME), Darwin)
    # Homebrew prefix for the *native* arch: /opt/homebrew on arm64.
    ifeq ($(shell uname -m), arm64)
        GSL_PREFIX ?= /opt/homebrew
    else
        GSL_PREFIX ?= /usr/local
    endif
endif

LIB_DIR := $(CURDIR)/lib
NATIVE   := $(CURDIR)/../../native

CFLAGS = -Wall -fPIC -I$(GSL_PREFIX)/include
# -J alongside -I: without it nifco2's own .mod lands next to the source
FFLAGS = -fPIC -I$(LIB_DIR) -J$(LIB_DIR)
LDFLAGS = -L$(GSL_PREFIX)/lib -L$(LIB_DIR) -Wl,-rpath,@loader_path
LDLIBS = -lgsl -lgslcblas -lgfortran -lminpack

C_SOURCES = $(CURDIR)/RCM_solv.c
FORTRAN_SOURCES = $(NATIVE)/nifco2.f90
C_OBJS = $(C_SOURCES:.c=.o)
FORTRAN_OBJS = $(FORTRAN_SOURCES:.f90=.o)

TARGET = $(LIB_DIR)/RCM_solver.so

# minpack is built from source rather than shipped prebuilt: a .so and its .mod
# are both compiler- and arch-specific, so a binary that matches the Python
# doing the dlopen is the only one worth having.
MINPACK_SRC = $(NATIVE)/minpack.f90
MINPACK_LIB = $(LIB_DIR)/libminpack.so

all: $(TARGET)

$(TARGET): $(C_OBJS) $(FORTRAN_OBJS) $(MINPACK_LIB)
	$(FC) -shared -o $@ $(C_OBJS) $(FORTRAN_OBJS) $(LDFLAGS) $(LDLIBS)

# -J puts minpack_module.mod where FFLAGS' -I will look for it
$(MINPACK_LIB): $(MINPACK_SRC)
	@mkdir -p $(LIB_DIR)
	$(FC) -fPIC -shared -J$(LIB_DIR) -o $@ $<

# nifco2.f90 does `use minpack_module`, so the .mod must exist first
$(FORTRAN_OBJS): $(MINPACK_LIB)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.f90
	$(FC) -fPIC -c $< -o $@ $(FFLAGS)

clean:
	rm -f $(C_OBJS) $(FORTRAN_OBJS) $(TARGET) $(MINPACK_LIB) $(LIB_DIR)/*.mod

.PHONY: all clean
