all: test

src/test_headers.cc: FORCE
	find include \
	  | grep '\.hpp' \
	  | sort \
	  | sed 's/include\/\(.*\)/#include <\1>/g' \
	  > src/test_headers.cc
	echo 'int main () { return 0; }' >> src/test_headers.cc

build_cc: src/test_headers.cc FORCE
	mkdir -p build lib
	cd build \
	  && cmake -DCMAKE_INSTALL_PREFIX=.. .. \
	  && $(MAKE) install

install_cc: build_cc FORCE
	test -e $(VIRTUAL_ENV) \
	  && cp lib/* $(VIRTUAL_ENV)/lib/

install_py: install_cc FORCE
	pip install -r requirements.txt
	pip install -e .

install: install_py install_cc

test: install
	pyflakes setup.py distributions derivations
	pep8 --repeat --exclude=*_pb2.py setup.py distributions derivations
	nosetests -v
	cd build && ctest
	./test_cmake.sh
	@echo '----------------'
	@echo 'PASSED ALL TESTS'

protobuf: FORCE
	protoc --cpp_out=include/ --python_out=. distributions/schema.proto
	mv include/distributions/schema.pb.cc src/
	@pyflakes distributions/schema_pb2.py \
	  || (echo '...patching schema_pb2.py' \
	    ; sed -i '/descriptor_pb2/d' distributions/schema_pb2.py)  # HACK

clean: FORCE
	git clean -Xdf

FORCE:
