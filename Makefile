PKG = imfind

build:
	pip install build
	python -m build

install: build
	pip install dist/*.tar.gz

develop:
	pip install -e .

check:
	pytest -v tests

uninstall:
	pip uninstall $(PKG)

clean:
	rm -rvf dist/ build/ src/*.egg-info

push-test:
	python -m twine upload --repository testpypi dist/*

pull-test:
	pip install -i https://test.pypi.org/simple/ $(PKG)

push-prod:
	python -m twine upload dist/*

pull-prod:
	pip install $(PKG)
