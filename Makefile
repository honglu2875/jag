SHELL=/bin/bash

install:
	pip install .

dev:
	pip install -e .[dev]

test:
	pytest tests/

style:
	black src/ tests/
	isort src/ tests/

.PHONY: 
	test
