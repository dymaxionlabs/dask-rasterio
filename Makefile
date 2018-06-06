.PHONY: test cov

cov:
	poetry run coverage run -m pytest
	poetry run coverage report
	poetry run coverage html

test:
	poetry run pytest