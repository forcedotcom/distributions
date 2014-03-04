all: test

install_cc: FORCE
	mkdir -p build
	cd build &&\
	  cmake -DCMAKE_INSTALL_PREFIX=$(VIRTUAL_ENV) .. &&\
	  $(MAKE) install

install_py: install_cc FORCE
	pip install -r requirements.txt
	pip install -e .

install: install_py install_cc

test: install
	pyflakes setup.py distributions
	pep8 --repeat --exclude=*_pb2.py setup.py distributions
	nosetests -v
	cd build && ctest
	./test_cmake.sh
	@echo '----------------'
	@echo 'PASSED ALL TESTS'

protobuf: FORCE
	protoc --cpp_out=include/ --python_out=. distributions/schema.proto
	mv include/distributions/schema.pb.cc src/
	@pyflakes distributions/schema_pb2.py ||\
	  echo '...patching schema_pb2.py' ;\
	  sed -i '/descriptor_pb2/d' distributions/schema_pb2.py  # HACK

clean: FORCE
	git clean -Xdf

FORCE:
