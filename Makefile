.PHONY: test cov

cov:
	RIO_IGNORE_CREATION_KWDS=TRUE poetry run coverage run -m pytest
	poetry run coverage report
	poetry run coverage html

test:
	RIO_IGNORE_CREATION_KWDS=TRUE poetry run pytest