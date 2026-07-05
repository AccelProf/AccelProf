#include "py_frame.h"

#include <Python.h>
#include <frameobject.h>
#include <pybind11/pybind11.h>

#include <sstream>
#include <string>
#include <vector>

typedef struct PythonFrame {
    std::string file_name;
    std::string func_name;
    size_t func_first_lineno;
    size_t lineno;

    PythonFrame(const std::string &file_name, const std::string &func_name,
                size_t func_first_lineno, size_t lineno)
            : file_name(file_name),
            func_name(func_name),
            func_first_lineno(func_first_lineno),
            lineno(lineno) {}
}PythonFrame_t;

class PyFrameChecker {
public:
    // Return the current python frames with a query or using the previous cached frames
    std::vector<PythonFrame_t> &get_frames(bool cached = false);

    // Get the singleton instance
    static PyFrameChecker &instance();

private:
    PyFrameChecker() {}

    std::string unpack_pyobject(PyObject *obj);

private:
    // Cached frames for each thread
    static inline thread_local std::vector<PythonFrame_t> _frames;
};



PyFrameChecker& PyFrameChecker::instance() {
    static PyFrameChecker monitor;
    return monitor;
}

// Take from PyTorch::THPUtils_unpackStringView
std::string PyFrameChecker::unpack_pyobject(PyObject* obj) {
    if (PyBytes_Check(obj)) {
        size_t size = PyBytes_GET_SIZE(obj);
        return std::string(PyBytes_AS_STRING(obj), size);
    }
    if (PyUnicode_Check(obj)) {
        // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
        Py_ssize_t size;
        const char* data = PyUnicode_AsUTF8AndSize(obj, &size);
        if (!data) {
            // If we get any runtime error, just return an empty string to continue running
            // printf("obj %p utf8 parsing error", obj);
            return "";
        }
        return std::string(data, (size_t)size);
    }
    // printf("obj %p not bytes or unicode", obj);
    return "";
}

std::vector<PythonFrame_t>& PyFrameChecker::get_frames(bool cached) {
    if (cached) {
        return _frames;
    }

    // GIL lock is required
    pybind11::gil_scoped_acquire gil;

    PyFrameObject* frame = PyEval_GetFrame();
    _frames.clear();

    while (nullptr != frame) {
        size_t lineno = (size_t)PyFrame_GetLineNumber(frame);

        #if PY_VERSION_HEX >= 0x030B0000
            // PyFrameObject is opaque in 3.11+, use accessors
            PyCodeObject* code = PyFrame_GetCode(frame);  // new ref (3.11+)
            size_t func_first_lineno = 0;
            std::string file_name, func_name;
            if (code) {
                PyObject* co_filename = PyObject_GetAttrString((PyObject*)code, "co_filename");
                PyObject* co_name = PyObject_GetAttrString((PyObject*)code, "co_name");
                PyObject* co_firstlineno = PyObject_GetAttrString((PyObject*)code, "co_firstlineno");
                file_name = co_filename ? unpack_pyobject(co_filename) : "";
                func_name = co_name ? unpack_pyobject(co_name) : "";
                if (co_firstlineno && PyLong_Check(co_firstlineno)) {
                    func_first_lineno = (size_t)PyLong_AsUnsignedLong(co_firstlineno);
                }
                Py_XDECREF(co_filename);
                Py_XDECREF(co_name);
                Py_XDECREF(co_firstlineno);
                Py_DECREF(code);
            }
            _frames.emplace_back(PythonFrame_t{file_name, func_name, func_first_lineno, lineno});
            frame = PyFrame_GetBack(frame);  // borrowed; do not DECREF
        #else
            // ≤3.10: direct field access still allowed
            size_t func_first_lineno = (size_t)frame->f_code->co_firstlineno;
            std::string file_name = unpack_pyobject(frame->f_code->co_filename);
            std::string func_name = unpack_pyobject(frame->f_code->co_name);
            _frames.emplace_back(PythonFrame_t{file_name, func_name, func_first_lineno, lineno});
            frame = frame->f_back;
        #endif
    }
    return _frames;
}

bool get_python_frames(std::vector<PythonFrame_t> &frames) {
    auto &frame_checker = PyFrameChecker::instance();
    frames = frame_checker.get_frames();    

    return frames.size() > 0;
}

std::string print_pyframes(int verbose) {
    std::vector<PythonFrame_t> python_frames;
    get_python_frames(python_frames);

    std::stringstream ss;
    for (size_t i = 0; i < python_frames.size(); i++) {
        ss << "f-" << std::to_string(i) << " "
           << std::string(python_frames[i].file_name) << ":"
           << std::to_string(python_frames[i].lineno) << "  def "
           << std::string(python_frames[i].func_name) << "() "
           << std::string(python_frames[i].file_name) << ":"
           << std::to_string(python_frames[i].func_first_lineno)
           << std::endl;
    }

    if (verbose) {
        printf("%s", ss.str().c_str());
        fflush(stdout);
    }

    return ss.str();
}


std::vector<std::string> get_pyframes(int keep) {
    std::vector<PythonFrame_t> python_frames;
    get_python_frames(python_frames);

    std::vector<std::string> frames;
    for (size_t i = 0; i < python_frames.size(); i++) {
        if (keep >= 0 && i >= keep) {
            break;
        }

        std::stringstream ss;
        ss << "f-" << std::to_string(i) << " "
           << std::string(python_frames[i].file_name) << ":"
           << std::to_string(python_frames[i].lineno) << "  def "
           << std::string(python_frames[i].func_name) << "()";

        frames.push_back(ss.str());
    }

    return frames;
}
