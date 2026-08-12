#define PY_SSIZE_T_CLEAN
#include <Python.h>

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "gil_restoring",
    "Test extension that intentionally omits a free-threading declaration.",
    -1,
    NULL,
};

PyMODINIT_FUNC PyInit_gil_restoring(void) {
    return PyModule_Create(&module);
}
