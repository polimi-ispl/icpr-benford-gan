#include "python-version.h"
#include <string.h>
#include "jpeg.h"

typedef struct {
  PyObject_HEAD
  struct jpeg_stuff stuff;
  char inited;
} py_JpegObject;

static void JpegObject_dealloc(py_JpegObject *self) {
  if (self->inited) {
    closejpeg(&self->stuff);
  }
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject *py_JpegObject_getComponentDimensions(py_JpegObject *self, PyObject *args) {
  int comp;
  if (!PyArg_ParseTuple(args, "i", &comp)) {
    return NULL;
  }
  if (comp < 0 || comp >= self->stuff.cinfo.num_components) {
    PyErr_SetString(PyExc_ValueError, "Component value out of range");
    return NULL;
  }
  jpeg_component_info *ji = self->stuff.cinfo.comp_info + comp;
  PyObject *result = Py_BuildValue("(ii)", ji->width_in_blocks, ji->height_in_blocks);
  Py_DECREF(args);
  return result;
}

static PyObject *py_JpegObject_setBlock(py_JpegObject *self, PyObject *args) {
  int x,y,comp;
  PyObject *coefObj;
  if (!PyArg_ParseTuple(args, "iiiO!", &x, &y, &comp, &PyByteArray_Type, &coefObj)) {
    return NULL;
  }
  Py_ssize_t size = PyByteArray_GET_SIZE(coefObj);
  if ((uint)size < BLOCKSIZE) {
    PyErr_SetString(PyExc_ValueError, "Byte array too small");
    return NULL;
  }
  JCOEFPTR ptr = get_coeff_block(&self->stuff, x, y, comp);
  if (ptr == NULL) {
    PyErr_SetString(PyExc_ValueError, "Coordinates or component out of range");
    return NULL;
  }
  char *coeffs = PyByteArray_AS_STRING(coefObj);
  memcpy(ptr, coeffs, BLOCKSIZE);
  Py_RETURN_NONE;
}

static PyObject *py_JpegObject_write(py_JpegObject *self, PyObject *args) {
  char *filename;
  if (!PyArg_ParseTuple(args, "s", &filename)) {
    return NULL;
  }
  FILE *file = fopen(filename, "wb");
  if (file == NULL) {
    PyErr_SetString(PyExc_ValueError, "Could not open file for writing");
    return NULL;
  }
  if (0 != writejpeg(file, &self->stuff)) {
    PyErr_SetString(PyExc_IOError, "Could not write jpeg");
    return NULL;
  }
  if (0 != fclose(file)) {
    PyErr_SetFromErrno(PyExc_IOError);
    return NULL;
  }
  Py_RETURN_NONE;
}



static PyObject *py_JpegObject_getBlock(py_JpegObject *self, PyObject *args) {
  int x,y, comp;
  if (!PyArg_ParseTuple(args, "iii", &x, &y, &comp)) {
    return NULL;
  }
  JCOEFPTR ptr = get_coeff_block(&self->stuff, x, y, comp);
  if (ptr == NULL) {
    PyErr_SetString(PyExc_ValueError, "Coordinates or component out of range");
    return NULL;
  }
  PyObject *bytes = PyByteArray_FromStringAndSize((const char*)ptr, BLOCKSIZE);

  Py_DECREF(args);

  return bytes;
}
  

static int py_JpegObject_init(py_JpegObject *self, PyObject *args, PyObject *kwds) {
  if (args == NULL) {
    return -1;
  }
  const char *filename;
  if (!PyArg_ParseTuple(args, "s", &filename)) {
    return -1;
  }
  FILE *file = fopen(filename, "rb");
  if (file == NULL) {
    PyErr_SetString(PyExc_IOError, "no such file!");
    return -1;
  }
  int err = FALSE;
  if (0 != readjpeg(file, &self->stuff)) {
    PyErr_SetString(PyExc_IOError, "file loading failed!");
    err = TRUE;
  } else {
    self->inited = TRUE;
  }
  if (0 != fclose(file)) {
    if (PyErr_Occurred() == NULL) {
      PyErr_SetFromErrno(PyExc_IOError);
    }
    err = TRUE;
  }
  if (err) {
    if (self->inited) {
      closejpeg(&self->stuff);
    }
    return -1;
  }
  return 0;
}

static PyMethodDef JpegObjectMethods[] = {
  {"getblock", (PyCFunction)py_JpegObject_getBlock, METH_VARARGS, "get a block"},
  {"getcomponentdimensions", (PyCFunction)py_JpegObject_getComponentDimensions, METH_VARARGS, "get component dimensions"},
  {"setblock", (PyCFunction)py_JpegObject_setBlock, METH_VARARGS, "set a block"},
  {"write", (PyCFunction)py_JpegObject_write, METH_VARARGS, "write to file"},
  {NULL, NULL, 0, NULL}
};

static PyObject *py_JpegObject_getComponentCount(py_JpegObject *self, void *closure) {
  return PyLong_FromLong(self->stuff.cinfo.num_components);
}

static PyGetSetDef JpegObjectGetSet[] = {
  {"component_count", (getter)py_JpegObject_getComponentCount, NULL, "number of components", NULL},
  {NULL, NULL, NULL, NULL, NULL}
};

static PyTypeObject py_JpegType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "jpeg.Jpeg",               /*tp_name*/
    sizeof(py_JpegObject),     /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)JpegObject_dealloc, /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    0,                         /*tp_repr*/
    0,                         /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    0,                         /*tp_as_mapping*/
    0,                         /*tp_hash */
    0,                         /*tp_call*/
    0,                         /*tp_str*/
    0,                         /*tp_getattro*/
    0,                         /*tp_setattro*/
    0,                         /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT,        /*tp_flags*/
    "jpeg objects",            /* tp_doc */
    0,		               /* tp_traverse */
    0,		               /* tp_clear */
    0,		               /* tp_richcompare */
    0,		               /* tp_weaklistoffset */
    0,		               /* tp_iter */
    0,		               /* tp_iternext */
    JpegObjectMethods,             /* tp_methods */
    0,             /* tp_members */
    JpegObjectGetSet,          /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)py_JpegObject_init, /* tp_init */
    0,                         /* tp_alloc */
    0,                 /* tp_new */
};

static PyMethodDef JpegMethods[] = {
  {NULL, NULL, 0, NULL}
};

#if python_version == 3
static PyModuleDef JpegModule = {
  PyModuleDef_HEAD_INIT,
  "jpeg",
  "libjpeg bindings",
  -1,
  JpegMethods, NULL, NULL, NULL, NULL
};
#endif

PyObject* doinit(void) {
  py_JpegType.tp_new = PyType_GenericNew;
  if (PyType_Ready(&py_JpegType) < 0) {
    return NULL;
  }
#if python_version == 3
  PyObject *m = PyModule_Create(&JpegModule);
#elif python_version == 2
  PyObject *m = Py_InitModule3("jpeg", JpegMethods, "hello world");
#endif
  if (m == NULL) {
    return NULL;
  }

  Py_INCREF(&py_JpegType);
  PyModule_AddObject(m, "Jpeg", (PyObject *)&py_JpegType);
  
  return m;
}
#if python_version == 3
PyMODINIT_FUNC PyInit_jpeg(void) {
  return doinit();
}
#elif python_version == 2
PyMODINIT_FUNC initjpeg(void) {
  doinit();
}
#endif
