PKG = imfind
PYTHON = python3
PIP = pip3

build:
	$(PIP) install build
	$(PYTHON) -m build

install: build
	$(PIP) install dist/*.tar.gz

develop:
	$(PIP) install -e .

check:
	pytest -v tests

uninstall:
	$(PIP) uninstall $(PKG)

clean:
	rm -rvf dist/ build/ src/*.egg-info

push-test:
	$(PYTHON) -m twine upload --repository testpypi dist/*

pull-test:
	$(PIP) install -i https://test.pypi.org/simple/ $(PKG)

push-prod:
	$(PYTHON) -m twine upload dist/*

pull-prod:
	$(PIP) install $(PKG)
