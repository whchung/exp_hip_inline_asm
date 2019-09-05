HIP_PATH?= $(wildcard /opt/rocm/hip)
ifeq (,$(HIP_PATH))
	HIP_PATH=../../..
endif
HIP_PLATFORM=$(shell $(HIP_PATH)/bin/hipconfig --platform)
HIPCC=$(HIP_PATH)/bin/hipcc

SOURCES=plus1.cpp

all: plus1.out

# Step
plus1.out: $(SOURCES)
	KMDUMPLLVM=1 KMDUMPISA=1 $(HIPCC) $(CXXFLAGS) $(SOURCES) -o $@

clean:
	rm -f dump* *.o *.out
