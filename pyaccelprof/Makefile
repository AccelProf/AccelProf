.PHONY: all install clean

all: install

install:
	python3 setup.py bdist
	@echo ""
	@echo "The package has been built successfully."
	@echo "To install the package, run the following command:"
	@echo "pip install ."

clean:
	rm -rf build dist *.egg-info

accelprof:
	cd .. && ./bin/build

accelprof_clean:
	cd .. && ./bin/clean
