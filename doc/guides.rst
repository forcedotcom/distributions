Guides
======

Write a Component Model
-----------------------

In this guide, you'll implement the Beta Bernoulli (BB) component
model.

First, build distributions according to the :ref:`developer quick
start instructions <installation-developer>`.

C++
^^^

* Create the header ``include/distributions/models/bb.hpp`` by copying
  ``include/distributions/models/nich.hpp``.
* Create the implementation ``src/models/bb.cc`` by copying
  ``src/models/nich.cc``.
* In both the header and the implementation, replace
  ``NormalInverseChiSq`` with ``BetaBernoulli``.
* In the implementation, replace ``nich.hpp`` with ``bb.hpp``.
* Add ``models/bb.cc`` to ``DISTRIBUTIONS_SOURCE_FILES`` in
  ``src/CMakeLists.txt``.
* Run ``make test_cc`` to ensure your new model compiles. This
  automatically detects your new header and builds it to ensure it
  compiles.

Wrap C++ for Python
^^^^^^^^^^^^^^^^^^^

* Create the cython wrapper ``distributions/lp/models/bb.pyx`` by
  copying ``distributions/lp/models/nich.pyx``.
* In ``bb.pyx``, replace ``NormalInverseChiSq`` with ``BetaBernoulli``
  and ``nich.hpp`` with ``bb.hpp``.
* Add ``lp.models.bb`` to the ``lp_extensions`` list in ``setup.py``.
* Run ``PYDISTRIBUTIONS_USE_LIB=1 make test_cy`` to ensure your new
  model compiles.

Now go update your component model fields and wonder why this wasn't
all templated out for you.
