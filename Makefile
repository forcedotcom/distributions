uname:=$(shell uname -s)
cpu_count=$(shell python -c 'import multiprocessing as m; print m.cpu_count()')


ld_library_path=
ifeq ($(uname),Linux)
	ld_library_path=LD_LIBRARY_PATH
endif
ifeq ($(uname),Darwin)
	ld_library_path=DYLD_LIBRARY_PATH
endif

cmake_args=
nose_env:=NOSE_PROCESSES=$(cpu_count) NOSE_PROCESS_TIMEOUT=240
ifdef VIRTUAL_ENV
	cmake_args=-DCMAKE_INSTALL_PREFIX=$(VIRTUAL_ENV)
	library_path=$(LIBRARY_PATH):$(VIRTUAL_ENV)/lib/
	nose_env+=$(ld_library_path)=$($(ld_library_path)):$(VIRTUAL_ENV)/lib/
else
	cmake_args=-DCMAKE_INSTALL_PREFIX=..
	library_path=$(LIBRARY_PATH):`pwd`/lib/
	nose_env+=$(ld_library_path)=$($(ld_library_path)):`pwd`/lib/
endif

cy_deps=
ifdef PYDISTRIBUTIONS_USE_LIB
	install_cy_deps=install_cc
endif

all: test

headers:=$(shell find include | grep '\.hpp' | grep -v protobuf | sort -d)
src/test_headers.cc: $(headers)
	echo $(headers) \
	  | sed 's/include\/\(\S*\)\s*/#include <\1>\n/g' \
	  > src/test_headers.cc
	echo 'int main () { return 0; }' >> src/test_headers.cc

configure_cc: src/test_headers.cc FORCE
	mkdir -p build lib
	cd build && cmake $(cmake_args) ..

build_cc: configure_cc FORCE
	cd build && $(MAKE)

install_cc: build_cc FORCE
	cd build && $(MAKE) install

install_cy: $(install_cy_deps) FORCE
	pip install -r requirements.txt
	LIBRARY_PATH=$(library_path) pip install -e .

install: install_cc install_cy FORCE

install_cc_examples: install_cc FORCE

test_cc_examples: install_cc_examples FORCE
	./test_cmake.sh
	@echo '----------------'
	@echo 'PASSED CC EXAMPLES'

test_cc: install_cc FORCE
	cd build && ctest
	@echo '----------------'
	@echo 'PASSED CC TESTS'

PY_SOURCES=setup.py update_license.py distributions derivations examples/mixture

test_cy: install_cy FORCE
	pyflakes $(PY_SOURCES)
	pep8 --repeat --ignore=E265 --exclude=*_pb2.py $(PY_SOURCES)
	$(nose_env) nosetests -v distributions derivations examples
	@echo '----------------'
	@echo 'PASSED CY TESTS'

test: test_cc test_cc_examples test_cy FORCE
	@echo '----------------'
	@echo 'PASSED ALL TESTS'

protobuf: FORCE
	protoc --cpp_out=include --python_out=. distributions/io/schema.proto
	mkdir -p src/io && cp include/distributions/io/schema.pb.cc src/io/
	@pyflakes distributions/io/schema_pb2.py \
	  || (echo '...patching schema_pb2.py' \
	    ; sed -i '/descriptor_pb2/d' distributions/io/schema_pb2.py)  # HACK

profile: install_cc FORCE
	build/benchmarks/sample_from_scores
	build/benchmarks/score_counts
	build/benchmarks/sample_assignment_from_py
	build/benchmarks/special
	build/benchmarks/mixture

profile_test: install
	nosetests --with-profile --profile-stats-file=nosetests.profile
	cat nosetests.profile

clean: FORCE
	git clean -Xdf

FORCE:
