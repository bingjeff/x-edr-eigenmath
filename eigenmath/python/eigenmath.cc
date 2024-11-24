#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>

#include "absl/strings/str_cat.h"
#include "eigenmath/manifolds.h"
#include "eigenmath/pose3.h"
#include "eigenmath/pose3_utils.h"
#include "eigenmath/types.h"

namespace pybind11::detail {

template <typename Scalar>
struct type_caster<eigenmath::Quaternion<Scalar>> {
 public:
  PYBIND11_TYPE_CASTER(eigenmath::Quaternion<Scalar>, _("Quaternion"));

  // Convert Python to C++.
  bool load(handle src, bool convert) {
    if (!convert) {
      return false;
    }
    auto buffer = array::ensure(src);
    if (!buffer || buffer.ndim() != 1 || buffer.size() != 4) {
      return false;
    }
    array_t<Scalar> array = src.cast<array_t<Scalar>>();
    auto reader = array.unchecked();
    value = eigenmath::Quaternion<Scalar>(/*w=*/reader(0), /*x=*/reader(1),
                                          /*y=*/reader(2), /*z=*/reader(3));
    return true;
  }

  // Convert C++ to Python.
  static handle cast(const eigenmath::Quaternion<Scalar>& src,
                     return_value_policy, handle) {
    array_t<Scalar> array({4});
    auto writer = array.mutable_unchecked();
    writer(0) = src.w();
    writer(1) = src.x();
    writer(2) = src.y();
    writer(3) = src.z();
    return array.release();
  }
};

}  // namespace pybind11::detail

namespace eigenmath {

namespace py = ::pybind11;

PYBIND11_MODULE(eigenmath, py_module) {
  py_module.doc() = "A module wrapping key eigenmath types and methods.";

  py::class_<Pose3d>(py_module, "Pose3d")
      .def(py::init<>())
      .def(py::init<const Vector3d&>())
      .def(py::init<const Quaterniond&>())
      .def(py::init<const Matrix3d&>())
      .def(py::init<const Matrix4d&>())
      .def(py::init<const Quaterniond&, const Vector3d&>())
      .def(py::init<const Matrix3d&, const Vector3d&>())
      .def(py::self *= py::self)
      .def(py::self * py::self)
      .def("__mul__", [](const Pose3d& self,
                         const Vector3d& point) { return self * point; })
      .def("__mul__",
           [](const Pose3d& self, const MatrixXd& points_3n) {
             if (points_3n.rows() != 3) {
               throw py::value_error(absl::StrCat(
                   "Row size should be 3, but got rows=", points_3n.rows()));
             }
             MatrixXd result = self.rotationMatrix() * points_3n.topRows<3>();
             result.row(0).array() += self.translation().x();
             result.row(1).array() += self.translation().y();
             result.row(2).array() += self.translation().z();
             return result;
           })
      .def_property_readonly("matrix",
                             py::overload_cast<>(&Pose3d::matrix, py::const_))
      .def_property(
          "translation", py::overload_cast<>(&Pose3d::translation, py::const_),
          [](Pose3d& self, Vector3d& vector) { self.translation() = vector; })
      .def_property("quaternion_wxyz",
                    py::overload_cast<>(&Pose3d::quaternion, py::const_),
                    [](Pose3d& self, Quaterniond& quaternion_wxyz) {
                      self.setQuaternion(quaternion_wxyz);
                    })
      .def_property(
          "rotation_vector", [](Pose3d& self) { return LogSO3(self.so3()); },
          [](Pose3d& self, Vector3d vector) {
            self.setQuaternion(ExpSO3(vector).quaternion());
          })
      .def_property("rotation_matrix",
                    py::overload_cast<>(&Pose3d::rotationMatrix, py::const_),
                    [](Pose3d& self, const Matrix3d& matrix3x3) {
                      self.setRotationMatrix(matrix3x3);
                    })
      .def_property_readonly("x_axis",
                             py::overload_cast<>(&Pose3d::xAxis, py::const_))
      .def_property_readonly("y_axis",
                             py::overload_cast<>(&Pose3d::yAxis, py::const_))
      .def_property_readonly("z_axis",
                             py::overload_cast<>(&Pose3d::zAxis, py::const_))
      .def("inverse", &Pose3d::inverse);

  py_module.def("RotationVector",
                [](Vector3d vec) { return Pose3d(ExpSO3(vec)); });
  py_module.def("RotationX", [](double theta) { return RotationX(theta); });
  py_module.def("RotationY", [](double theta) { return RotationY(theta); });
  py_module.def("RotationZ", [](double theta) { return RotationZ(theta); });
}
}  // namespace eigenmath
