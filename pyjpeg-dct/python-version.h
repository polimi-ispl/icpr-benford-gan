#define python_version 3

#if python_version == 3
#include <python3.6m/Python.h>
#elif python_version == 2
#include <python2.7/Python.h>
#endif
