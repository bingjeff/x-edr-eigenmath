// Copyright 2023 Google LLC

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     https://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef EIGENMATH_EIGENMATH_POSE3_UTILS_H_
#define EIGENMATH_EIGENMATH_POSE3_UTILS_H_

#include "Eigen/Core"
#include "absl/types/optional.h"
#include "manifolds.h"
#include "pose2.h"
#include "pose3.h"
#include "rotation_utils.h"
#include "types.h"
#include "utils.h"

namespace eigenmath {

// Generates rotational pose about x-axis
template <class T>
Pose3<T> RotationX(T theta) {
  return Pose3<T>(ExpSO3(Vector3<T>(theta, T(0.), T(0.))));
}

// Generates rotational pose about y-axis
template <class T>
Pose3<T> RotationY(T theta) {
  return Pose3<T>(ExpSO3(Vector3<T>(T(0.), theta, T(0.))));
}

// Generates rotational pose about z-axis
template <class T>
Pose3<T> RotationZ(T theta) {
  return Pose3<T>(ExpSO3(Vector3<T>(T(0.), T(0.), theta)));
}

namespace pose_conversion_details {

// Extracts 2x2 submatrices corresponding to the rotational component
// about x.
// Note: The returned matrix is not necessarily a valid 2x2 rotation matrix.
template <class T, int Options = kDefaultOptions>
Matrix2<T> GetRotationSubMatrixX(const SO3<T, Options>& rotation) {
  return rotation.matrix().template bottomRightCorner<2, 2>();
}

// Extracts 2x2 submatrices corresponding to the rotational component
// about y.
// Note: The returned matrix is not necessarily a valid 2x2 rotation matrix.
template <class T, int Options = kDefaultOptions>
Matrix2<T> GetRotationSubMatrixY(const SO3<T, Options>& rotation) {
  const Matrix3<T>& R = rotation.matrix();
  Matrix2<T> rotation_submatrix_about_y;
  // note: The order here, R(2,0) in first row and R(0,2) in the second row is
  // intentional. see
  // http://en.wikipedia.org/w/index.php?title=Rotation_matrix&oldid=628483834#Basic_rotations
  rotation_submatrix_about_y << R(0, 0), R(2, 0), R(0, 2), R(2, 2);
  return rotation_submatrix_about_y;
}

// Extracts 2x2 submatrices corresponding to the rotational component
// about z.
// Note: The returned matrix is not necessarily a valid 2x2 rotation matrix.
template <class T, int Options = kDefaultOptions>
Matrix2<T> GetRotationSubMatrixZ(const SO3<T, Options>& rotation) {
  return rotation.matrix().template topLeftCorner<2, 2>();
}

}  // namespace pose_conversion_details

// Extracts SO2 rotation instance about x-axis
// This method orthogonalizes the 2x2 sub-matrix for the x-axis rotation and
// extracts the corresponding two dimensional rotation.
template <class T, int Options = kDefaultOptions>
SO2<T> ToSO2X(const SO3<T, Options>& rotation) {
  Matrix2<T> rotation_about_x = OrthogonalizeRotationMatrix(
      pose_conversion_details::GetRotationSubMatrixX(rotation));
  return SO2<T>(rotation_about_x(0, 0), rotation_about_x(1, 0), false);
}

// Extracts SO2 rotation instance about x-axis
// This method orthogonalizes the 2x2 sub-matrix for the x-axis rotation and
// extracts the corresponding two dimensional rotation.
template <class T, int Options = kDefaultOptions>
SO2<T> ToSO2X(const Pose3<T, Options>& pose3) {
  return ToSO2X(pose3.so3());
}

// Extracts SO2 rotation instance about y-axis
// This method orthogonalizes the 2x2 sub-matrix for the y-axis rotation and
// extracts the corresponding two dimensional rotation.
template <class T, int Options = kDefaultOptions>
SO2<T> ToSO2Y(const SO3<T, Options>& rotation) {
  Matrix2<T> rotation_about_y = OrthogonalizeRotationMatrix(
      pose_conversion_details::GetRotationSubMatrixY(rotation));
  return SO2<T>(rotation_about_y(0, 0), rotation_about_y(1, 0), false);
}

// Extracts SO2 rotation instance about y-axis
// This method orthogonalizes the 2x2 sub-matrix for the y-axis rotation and
// extracts the corresponding two dimensional rotation.
template <class T, int Options = kDefaultOptions>
SO2<T> ToSO2Y(const Pose3<T, Options>& pose3) {
  return ToSO2Y(pose3.so3());
}

// Extracts SO2 rotation instance about z-axis
// This method orthogonalizes the 2x2 sub-matrix for the z-axis rotation and
// extracts the corresponding two dimensional rotation.
template <class T, int Options = kDefaultOptions>
SO2<T> ToSO2Z(const SO3<T, Options>& rotation) {
  Matrix2<T> rotation_about_z = OrthogonalizeRotationMatrix(
      pose_conversion_details::GetRotationSubMatrixZ(rotation));
  return SO2<T>(rotation_about_z(0, 0), rotation_about_z(1, 0), false);
}

// Extracts SO2 rotation instance about z-axis
// This method orthogonalizes the 2x2 sub-matrix for the x-axis rotation and
// extracts the corresponding two dimensional rotation.
template <class T, int Options = kDefaultOptions>
SO2<T> ToSO2Z(const Pose3<T, Options>& pose3) {
  return ToSO2Z(pose3.so3());
}

// Extracts rotation angle about x-axis
// This method orthogonalizes the 2x2 sub-matrix for the x-axis rotation and
// extracts the corresponding angle.
template <class T, int Options = kDefaultOptions>
T ToRotationAngleX(const SO3<T, Options>& rotation) {
  return ToSO2X(rotation).angle();
}

// Extracts rotation angle about x-axis
// This method orthogonalizes the 2x2 sub-matrix for the x-axis rotation and
// extracts the corresponding angle. In particular, it is the inverse of
// RotationX.
template <class T, int Options = kDefaultOptions>
T ToRotationAngleX(const Pose3<T, Options>& pose3) {
  return ToSO2X(pose3).angle();
}

// Extracts rotation angle about y-axis
// This method orthogonalizes the 2x2 sub-matrix for the y-axis rotation and
// extracts the corresponding angle.
template <class T, int Options = kDefaultOptions>
T ToRotationAngleY(const SO3<T, Options>& rotation) {
  return ToSO2Y(rotation).angle();
}

// Extracts rotation angle about y-axis
// This method orthogonalizes the 2x2 sub-matrix for the y-axis rotation and
// extracts the corresponding angle. In particular, it is the inverse of
// RotationY.
template <class T, int Options = kDefaultOptions>
T ToRotationAngleY(const Pose3<T, Options>& pose3) {
  return ToSO2Y(pose3).angle();
}

// Extracts rotation angle about z-axis
// This method orthogonalizes the 2x2 sub-matrix for the z-axis rotation and
// extracts the corresponding angle.
template <class T, int Options = kDefaultOptions>
T ToRotationAngleZ(const SO3<T, Options>& rotation) {
  return ToSO2Z(rotation).angle();
}

// Extracts rotation angle about z-axis
// This method orthogonalizes the 2x2 sub-matrix for the z-axis rotation and
// extracts the corresponding angle. In particular, it is the inverse of
// RotationZ.
template <class T, int Options = kDefaultOptions>
T ToRotationAngleZ(const Pose3<T, Options>& pose3) {
  return ToSO2Z(pose3).angle();
}

// Extracts pose in x/y plane
// Note: The 2d x-axis corresponds to the 3d x axis and the 2d y-axis
// corresponds to the 3d y-axis.
template <class T, int Options = kDefaultOptions>
Pose2<T> ToPose2XY(const Pose3<T, Options>& pose3) {
  return Pose2<T>(Vector2<T>(pose3.translation().template head<2>()),
                  ToSO2Z(pose3));
}

// Extracts pose in z/x plane
// Note: The 2d x-axis corresponds to the 3d z axis and the 2d y-axis
// corresponds to the 3d x-axis.
template <class T, int Options = kDefaultOptions>
Pose2<T> ToPose2ZX(const Pose3<T, Options>& pose3) {
  return Pose2<T>(Vector2<T>(pose3.translation().z(), pose3.translation().x()),
                  ToSO2Y(pose3));
}

// Extracts pose in y/z plane
// Note: The 2d x-axis corresponds to the 3d y axis and the 2d y-axis
// corresponds to the 3d z-axis.
template <class T, int Options = kDefaultOptions>
Pose2<T> ToPose2YZ(const Pose3<T, Options>& pose3) {
  return Pose2<T>(Vector2<T>(pose3.translation().template tail<2>()),
                  ToSO2X(pose3));
}

// Converts two dimensional pose in x/y plane to Pose3
template <class T, int Options = kDefaultOptions>
Pose3<T> FromPose2XY(const Pose2<T, Options>& pose2) {
  Pose3<T> pose3_xy(RotationZ(pose2.angle()));
  pose3_xy.translation().template head<2>() = pose2.translation();
  return pose3_xy;
}

// Converts two dimensional pose in z/x plane to Pose3
// Note: The 2d x-axis corresponds to the 3d z axis and the 2d y-axis
// corresponds to the 3d x-axis.
// pose2:   two dimensional pose in z/c plabe
// Returns three dimensional pose
template <class T, int Options = kDefaultOptions>
Pose3<T> FromPose2ZX(const Pose2<T, Options>& pose2) {
  Pose3<T> pose3_zx(RotationY(pose2.angle()));
  pose3_zx.translation().z() = pose2.translation().x();
  pose3_zx.translation().x() = pose2.translation().y();
  return pose3_zx;
}

// Converts two dimensional pose in y/z plane to Pose3
// Note: The 2d x-axis corresponds to the 3d y axis and the 2d y-axis
// corresponds to the 3d z-axis.
// pose2:   two dimensional pose in y/z plabe
// Returns three dimensional pose
template <class T, int Options = kDefaultOptions>
Pose3<T> FromPose2YZ(const Pose2<T, Options>& pose2) {
  Pose3<T> pose3_yz(RotationX(pose2.angle()));
  pose3_yz.translation().template tail<2>() = pose2.translation();
  return pose3_yz;
}

// Generates translational pose
// x: translation along x
// y: translation along y
// z: translation along z
// Returns Pose3 representing a translation
template <class Scalar>
Pose3<Scalar> Translation(Scalar x, Scalar y, Scalar z) {
  return Pose3<Scalar>(Vector3<Scalar>(x, y, z));
}

// Generates translational pose along x
// x: translation along x
// Returns Pose3 representing a translation along x
template <class Scalar>
Pose3<Scalar> TranslationX(Scalar x) {
  return Pose3<Scalar>(Vector3<Scalar>(x, Scalar(0), Scalar(0)));
}

// Generates translational pose along y
// y: translation along y
// Returns Pose3 representing a translation along y
template <class Scalar>
Pose3<Scalar> TranslationY(Scalar y) {
  return Pose3<Scalar>(Vector3<Scalar>(Scalar(0), y, Scalar(0)));
}

// Generates translational pose along z
// z: translation along z
// Returns Pose3 representing a translation along z
template <class Scalar>
Pose3<Scalar> TranslationZ(Scalar z) {
  return Pose3<Scalar>(Vector3<Scalar>(Scalar(0), Scalar(0), z));
}

// Translational residual between two poses
// reference_pose_a: one pose
// reference_pose_b: other pose
// Returns residual 3-vector of the translational difference in the reference
// frame
template <class T, int Options_a, int Options_b>
Vector3<T> TranslationResidual(const Pose3<T, Options_a>& reference_pose_a,
                               const Pose3<T, Options_b>& reference_pose_b) {
  return reference_pose_b.translation() - reference_pose_a.translation();
}

// Translational error between two poses
// reference_pose_a: one pose
// reference_pose_b: other pose
// Returns positional error
template <class T, int Options_a, int Options_b>
T TranslationError(const Pose3<T, Options_a>& reference_pose_a,
                   const Pose3<T, Options_b>& reference_pose_b) {
  Vector3<T> t_res = TranslationResidual(reference_pose_a, reference_pose_b);
  return t_res.norm();
}

// Rotational residual between two poses
// reference_pose_a: one pose
// reference_pose_b: other pose
// Returns residual 3-vector of the relative rotational difference between
// frame a and b
template <class T, int Options_a, int Options_b>
Vector3<T> RotationResidual(const Pose3<T, Options_a>& reference_pose_a,
                            const Pose3<T, Options_b>& reference_pose_b) {
  return LogSO3(reference_pose_a.so3().inverse() * reference_pose_b.so3());
}

// Rotational error between two poses
// reference_pose_a: one pose
// reference_pose_b: other pose
// Returns angular error
template <class T, int Options_a, int Options_b>
T RotationError(const Pose3<T, Options_a>& reference_pose_a,
                const Pose3<T, Options_b>& reference_pose_b) {
  Vector3<T> rot_res = RotationResidual(reference_pose_a, reference_pose_b);
  return rot_res.norm();
}

// Translational and rotational residual between two poses
// reference_pose_a: one pose
// reference_pose_b: other pose
// Returns residual 6-vector of the relative translational and rotational
// difference between frame a and b
template <class T, int Options_a, int Options_b>
Vector6<T> PoseResidual(const Pose3<T, Options_a>& reference_pose_a,
                        const Pose3<T, Options_b>& reference_pose_b) {
  Vector6<T> residual;
  residual.template head<3>() =
      TranslationResidual(reference_pose_a, reference_pose_b);
  residual.template tail<3>() =
      RotationResidual(reference_pose_a, reference_pose_b);
  return residual;
}

struct PoseError {
  double translation;
  double rotation;
};

// Calculates pose error between two poses
template <class T, int Options_a, int Options_b>
PoseError PoseErrorBetween(const Pose3<T, Options_a>& reference_pose_a,
                           const Pose3<T, Options_b>& reference_pose_b) {
  struct PoseError error;
  error.translation = TranslationError(reference_pose_a, reference_pose_b);
  error.rotation = RotationError(reference_pose_a, reference_pose_b);
  return error;
}

// Rotates the covariance from one frame to another.
// a_covariance: Covariance in frame a.
// b_rotation_a: 3D rotation.
// Returns a_covariance in frame b.
template <class Scalar, int Options_cov, int Options_so3>
Matrix3<Scalar> RotateCovariance(
    const Matrix3<Scalar, Options_cov>& a_covariance,
    const SO3<Scalar, Options_so3>& b_rotation_a) {
  const eigenmath::Matrix3d transform = b_rotation_a.matrix();
  return transform * a_covariance * transform.transpose();
}

// Rotates the covariance from one frame to another.
// a_covariance: Covariance in frame a.
// b_rotation_a: 3D rotation.
// Returns a_covariance in frame b.
template <class Scalar, int Options_cov, int Options_so3>
Matrix6<Scalar> RotateCovariance(
    const Matrix6<Scalar, Options_cov>& a_covariance,
    const SO3<Scalar, Options_so3>& b_rotation_a) {
  Matrix6<Scalar> transform;
  transform.template block<3, 3>(0, 0) = transform.template block<3, 3>(3, 3) =
      b_rotation_a.matrix();
  transform.template block<3, 3>(0, 3) = transform.template block<3, 3>(3, 0) =
      Matrix3<Scalar>::Zero();
  return transform * a_covariance * transform.transpose();
}

// Calculates SE(3) adjoint.
// Please refer to http://ethaneade.com/lie.pdf for the proof.
// pose: SE(3) element
// Returns adjoint of given `pose`
template <class Scalar, int Options>
Matrix6<Scalar> AdjointSE3(const Pose3<Scalar, Options>& pose) {
  Matrix6<Scalar> adjoint_SE3;
  adjoint_SE3.template block<3, 3>(0, 0) = pose.rotationMatrix();
  adjoint_SE3.template block<3, 3>(0, 3) =
      SkewMatrix(pose.translation()) * pose.rotationMatrix();
  adjoint_SE3.template block<3, 3>(3, 0).setZero();
  adjoint_SE3.template block<3, 3>(3, 3) = pose.rotationMatrix();
  return adjoint_SE3;
}

// Transforms `y_pose_z_covariance` to `x_pose_z_covariance` assuming that
// `y_pose_z_covariance` is the covariance of `y_pose_z` perturbed on the left.
template <class Scalar, int Options_pose, int Options_cov>
Matrix6<Scalar> TransformCovariance(
    const Pose3<Scalar, Options_pose>& x_pose_y,
    const Matrix6<Scalar, Options_cov>& y_pose_z_covariance) {
  const Matrix6<Scalar> adjoint_x_pose_y = AdjointSE3(x_pose_y);
  return adjoint_x_pose_y * y_pose_z_covariance * adjoint_x_pose_y.transpose();
}

}  // namespace eigenmath

#endif  // EIGENMATH_EIGENMATH_POSE3_UTILS_H_
