install:
	pip install -e .

test:
	pip install -e .
	pyflakes setup.py distributions
	pep8 --repeat setup.py distributions
	nosetests -v
	./ctest.sh
	./test_cmake.sh

clean:
	git clean -dfx
