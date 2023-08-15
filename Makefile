SHELL=/bin/bash

install:
	pip install .

dev:
	pip install -e .[dev]

test:
	pytest tests/

.PHONY: 
	test
