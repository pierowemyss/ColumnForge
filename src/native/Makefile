# Builds the optional compiled thermo kernel: f2py wrapper around nifco2.f90.
#
#     make -C src/native            # -> src/native/lib/nifco.<abi>.so
#     make -C src/native clean
#
# The result is loaded by core/nifco.py and switched on in Preferences ->
# "Use NIFCO compiled thermo". Nothing in the app needs it; without it every
# thermo path stays on NumPy.
#
# PY must be the interpreter that will dlopen the result. On Apple silicon that
# is the /opt/homebrew python3.13, and gfortran has to match its arch — a .so
# built by the x86_64 Homebrew fails at dlopen, not at compile.
PY ?= python3.13
ifeq ($(origin FC), default)
    FC := gfortran
endif

LIB := $(CURDIR)/lib
EXT := $(shell $(PY) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
TARGET := $(LIB)/nifco$(EXT)

# minpack supplies hybrd1 for antoine_tsat / SRK_phi. It lives here beside its
# only caller (nifco2.f90) rather than in two places: 3800 lines of vendored
# solver copied twice is 3800 lines that can drift apart. The RCM Makefile
# links this same source.
MINPACK_SRC := $(CURDIR)/minpack.f90

all: $(TARGET)

# f2py's crackfortran cannot parse minpack.f90 (an array-constructor PARAMETER
# it gives up on), so minpack is compiled ahead of time and linked in as a
# static archive. -J puts minpack_module.mod where f2py's -I will find it.
$(LIB)/libminpack.a: $(MINPACK_SRC) | $(LIB)
	$(FC) -O2 -fPIC -J$(LIB) -c $< -o $(LIB)/minpack.o
	ar rcs $@ $(LIB)/minpack.o

# f2py's meson backend writes the extension into the current directory.
$(TARGET): nifco2.f90 $(LIB)/libminpack.a
	cd $(LIB) && $(PY) -m numpy.f2py -c $(CURDIR)/nifco2.f90 -m nifco \
		--backend meson --f90flags=-I$(LIB) -L$(LIB) -lminpack

$(LIB):
	mkdir -p $(LIB)

# Sweeps $(CURDIR) too: hand-run gfortran drops .o/.mod next to the source, and a
# stale .mod is how a wrong module interface gets picked up silently.
clean:
	rm -rf $(LIB)
	rm -f $(CURDIR)/*.o $(CURDIR)/*.mod

.PHONY: all clean
