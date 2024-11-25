load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//visibility:public"])

pybind_extension(
    name = "eigenmath",
    srcs = ["eigenmath.cc"],
    deps = [
        "//eigenmath",
        "@com_google_absl//absl/strings",
        "@eigen",
    ],
)

py_library(
    name = "eigenmath_lib",
    data = [":eigenmath"],
    imports = ["."],
)

py_test(
    name = "eigenmath_test",
    srcs = ["eigenmath_test.py"],
    deps = [
        ":eigenmath_lib",
        "@pypi//numpy",
    ],
)
