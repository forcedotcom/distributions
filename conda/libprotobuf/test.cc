#include "schema.pb.h"
#include <cassert>
int
main(int argc, char **argv)
{
  HelloWorld m;
  m.set_message("hi");
  assert( m.message() == "hi" );
  return 0;
}
