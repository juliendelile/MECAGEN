.PHONY: all isf mecagen cleanall

all: isf mecagen

isf:
	$(MAKE) -C ./isf

mecagen:
	$(MAKE) -C ./mecagen

cleanall:
	$(MAKE) cleanall -C ./isf
	$(MAKE) cleanall -C ./mecagen