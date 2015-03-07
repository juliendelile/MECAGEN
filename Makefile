.PHONY: all isf mecagen propper

all: isf mecagen

isf:
	$(MAKE) -C ./isf

mecagen:
	$(MAKE) -C ./mecagen

propper:
	$(MAKE) propper -C ./isf
	$(MAKE) propper -C ./mecagen