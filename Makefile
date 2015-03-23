uname:=$(shell uname -s)
cpu_count=$(shell python -c 'import multiprocessing as m; print m.cpu_count()')


ld_library_path=
sed_no_backup_arg=
ifeq ($(uname),Linux)
	ld_library_path=LD_LIBRARY_PATH
	sed_no_backup_arg=-i
endif
ifeq ($(uname),Darwin)
	ld_library_path=DYLD_LIBRARY_PATH
	sed_no_backup_arg=-i ''
endif

cmake_args=
nose_env:=NOSE_PROCESSES=$(cpu_count) NOSE_PROCESS_TIMEOUT=240
ifdef VIRTUAL_ENV
	root_path=$(VIRTUAL_ENV)
else
	ifdef CONDA_ROOT
		root_path=$(CONDA_ROOT)
	else
		root_path='../..'
	endif
endif
cmake_args=-DCMAKE_INSTALL_PREFIX=$(root_path)
library_path=$(LIBRARY_PATH):$(root_path)/lib/
nose_env+=$(ld_library_path)=$($(ld_library_path)):$(root_path)/lib/

ifdef CMAKE_INSTALL_PREFIX
	cmake_args=-DCMAKE_INSTALL_PREFIX=$(CMAKE_INSTALL_PREFIX)
endif

all: test

headers:=$(shell find include | grep '\.hpp' | grep -v protobuf | sort -d)
src/test_headers.cc: $(headers)
	find include | grep '\.hpp' | grep -v protobuf | sort -d \
	  | sed 's/include\/\(.*\)/#include <\1>/g' \
	  > src/test_headers.cc
	@echo '\nint main () { return 0; }' >> src/test_headers.cc

prepare_cc: src/test_headers.cc FORCE
	mkdir -p lib

build/python: prepare_cc FORCE
	mkdir -p build/python
	cd build/python && cmake -DCMAKE_BUILD_TYPE=Python $(cmake_args) ../..

build/debug: prepare_cc FORCE
	mkdir -p build/debug
	cd build/debug && cmake -DCMAKE_BUILD_TYPE=Debug $(cmake_args) ../..

build/release: prepare_cc FORCE
	mkdir -p build/release
	cd build/release && cmake -DCMAKE_BUILD_TYPE=Release $(cmake_args) ../..

python_cc: build/python FORCE
	cd build/python && $(MAKE)

debug_cc: build/debug FORCE
	cd build/debug && $(MAKE)

release_cc: build/release FORCE
	cd build/release && $(MAKE)

install_cc: python_cc debug_cc release_cc FORCE
	cd build/python && $(MAKE) install
	cd build/debug && $(MAKE) install
	cd build/release && $(MAKE) install

deps_cy: install_cc FORCE
	pip install -r requirements.txt

dev_cy: deps_cy FORCE
	LIBRARY_PATH=$(library_path) pip install -e .

install_cy: deps_cy FORCE
	LIBRARY_PATH=$(library_path) pip install --upgrade .

install: install_cc install_cy FORCE

package: python_cc debug_cc release_cc FORCE
	cd build && $(MAKE) package

install_cc_examples: install_cc FORCE

test_cc_examples: install_cc_examples FORCE
	$(ld_library_path)=$(library_path) ./test_cmake.sh
	@echo '----------------'
	@echo 'PASSED CC EXAMPLES'

CPP_SOURCES:=$(shell find include src examples benchmarks | grep -v 'vendor\|\.pb\.'  | grep -v 'src/test_headers.cc' | grep '\.\(cc\|hpp\)$$')

lint_cc: FORCE
	cpplint --filter=-build/include_order,-readability/streams,-readability/function,-runtime/arrays,-runtime/reference,-runtime/explicit,-readability/alt_tokens,-build/c++11 $(CPP_SOURCES)

test_cc: install_cc lint_cc FORCE
	cd build && ctest
	@echo '----------------'
	@echo 'PASSED CC TESTS'

PY_SOURCES=setup.py update_license.py distributions derivations examples/mixture

test_cy: dev_cy FORCE
	pyflakes $(PY_SOURCES)
	pep8 --repeat --ignore=E265,E402,W503 --exclude=*_pb2.py,vendor $(PY_SOURCES)
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
	    ; sed $(sed_no_backup_arg) '/descriptor_pb2/d' distributions/io/schema_pb2.py)  # HACK

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
