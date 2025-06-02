src := lazy_imports
test-src := tests
other-src := setup.py

check:
	pydocstyle --count $(src) $(other-src)
	black $(src) $(test-src) $(other-src) --check --diff
	flake8 $(src) $(other-src)
	isort $(src) $(test-src) $(other-src) --check --diff
	mdformat --check *.md
	mypy --install-types --non-interactive $(src) $(other-src)
	pylint $(src)

format:
	black $(src) $(test-src) $(other-src)
	isort $(src) $(test-src) $(other-src)
	mdformat *.md

test:
	pytest $(test-src)
