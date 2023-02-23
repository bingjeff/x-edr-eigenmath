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

#ifndef EIGENMATH_EIGENMATH_POSE2_H_
#define EIGENMATH_EIGENMATH_POSE2_H_

#include <ostream>

#include "Eigen/Core"  // IWYU pragma: keep
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "so2.h"
#include "types.h"
#include "utils.h"

namespace eigenmath {

// A pose in 2d space.
template <typename Scalar, int Options = kDefaultOptions>
class Pose2 {
 public:
  // Identity pose
  Pose2() : translation_(Scalar{0}, Scalar{0}) {}

  // Creates a pose with given translation and rotation
  template <int TranslationOptions = kDefaultOptions,
            int RotionationOptions = kDefaultOptions>
  constexpr Pose2(const Vector2<Scalar, TranslationOptions>& translation,
                  const SO2<Scalar, RotionationOptions>& rotation)
      : translation_{translation}, rotation_{rotation} {}

  // Creates a pose with given translation and rotation angle
  template <int OtherOptions = kDefaultOptions>
  Pose2(const Vector2<Scalar, OtherOptions>& translation, Scalar angle)
      : Pose2(translation, SO2<Scalar>{angle}) {}

  // Creates a pose from a rotation matrix and a translation vector
  template <int TranslationOptions = kDefaultOptions,
            int RotionationOptions = kDefaultOptions>
  Pose2(const Vector2<Scalar, TranslationOptions>& translation,
        const Matrix2<Scalar, RotionationOptions>& rotation_matrix)
      : translation_(translation), rotation_(rotation_matrix) {}

  // Creates a pose from a 3 x 3 affine transformation matrix
  explicit Pose2(const Matrix3<Scalar>& affine)
      : translation_(affine.template topRightCorner<2, 1>().eval()),
        rotation_(affine.template topLeftCorner<2, 2>().eval()) {}

  template <int OtherOptions>
  Pose2(const Pose2<Scalar, OtherOptions>& other)  // NOLINT
      : translation_{other.translation()}, rotation_{other.so2()} {}

  template <int OtherOptions>
  Pose2& operator=(const Pose2<Scalar, OtherOptions>& other) {
    translation_ = other.translation();
    rotation_ = other.so2();
    return *this;
  }

  static Pose2<Scalar> Identity() { return Pose2<Scalar>{}; }

  // Creates a pure translation
  static Pose2<Scalar> Translation(Scalar translation_x, Scalar translation_y) {
    return Pose2{{translation_x, translation_y}, SO2<Scalar>{}};
  }

  // Creates a pure translation
  template <typename Derived>
  static Pose2<Scalar> Translation(
      const Eigen::MatrixBase<Derived>& translation) {
    return Pose2<Scalar>{Vector2<Scalar>{translation}, SO2<Scalar>{}};
  }

  // Creates a pure rotation
  template <int OtherOptions>
  static Pose2<Scalar> Rotation(const SO2<Scalar, OtherOptions>& rotation) {
    return Pose2<Scalar>{{Scalar(0), Scalar(0)}, rotation};
  }

  // Creates a pure rotation
  static Pose2<Scalar> Rotation(Scalar angle) {
    return Pose2<Scalar>{{Scalar(0), Scalar(0)}, angle};
  }

  constexpr const Vector2<Scalar, Options>& translation() const {
    return translation_;
  }
  Vector2<Scalar, Options>& translation() { return translation_; }

  constexpr const SO2<Scalar, Options>& so2() const { return rotation_; }
  SO2<Scalar, Options>& so2() { return rotation_; }

  Scalar angle() const { return rotation_.angle(); }
  void setAngle(Scalar angle) { rotation_ = SO2<Scalar>{angle}; }

  // Normalizes the SO2 rotation representation
  void normalize() { rotation_.normalize(); }

  // Returns true if the underlying SO2 rotation representation is normalized.
  bool isNormalized() const { return rotation_.isNormalized(); }

  // Returns true if this pose is in a sane numeric state.
  bool isFinite() const {
    return std::isfinite(translation_.x()) && std::isfinite(translation_.y()) &&
           rotation_.isNormalized();
  }

  // Gives the 2D rotation matrix
  Matrix2<Scalar> rotationMatrix() const { return rotation_.matrix(); }

  // Sets the 2D rotation matrix
  template <int OtherOptions = kDefaultOptions>
  void setRotationMatrix(const Matrix2<Scalar, OtherOptions>& rotation_matrix) {
    rotation_ = SO2<Scalar, Options>(rotation_matrix);
  }

  // x-axis of the coordinate frame
  Vector2<Scalar> xAxis() const { return rotation_.xAxis(); }

  // y - axis of the coordinate frame
  Vector2<Scalar> yAxis() const { return rotation_.yAxis(); }

  constexpr Pose2<Scalar> inverse() const {
    SO2<Scalar> rotation_inverse = rotation_.inverse();
    Vector2<Scalar> translation_inverse = -(rotation_inverse * translation_);
    return Pose2<Scalar>{translation_inverse, rotation_inverse};
  }

  // Affine transformation matrix an element of R^3
  Matrix3<Scalar> matrix() const {
    Scalar cos_angle = rotation_.cos_angle();
    Scalar sin_angle = rotation_.sin_angle();
    return MakeMatrix<Scalar, 3, 3>({{cos_angle, -sin_angle, translation_.x()},
                                     {sin_angle, cos_angle, translation_.y()},
                                     {Scalar(0), Scalar(0), Scalar(1)}});
  }

  //  Cast Pose2 instance to other scalar type
  template <typename OtherScalar>
  Pose2<OtherScalar> cast() const {
    return Pose2<OtherScalar>(
        Vector2<OtherScalar>(translation_.template cast<OtherScalar>()),
        rotation_.template cast<OtherScalar>());
  }

  //  Checks if identical to another pose under a given tolerance
  template <int OtherOptions>
  bool isApprox(const Pose2<Scalar, OtherOptions>& other,
                Scalar linear_tolerance, Scalar angular_tolerance) const {
    return ((translation() - other.translation()).norm() < linear_tolerance) &&
           so2().isApprox(other.so2(), angular_tolerance);
  }

 private:
  Vector2<Scalar, Options> translation_;
  SO2<Scalar, Options> rotation_;
};

template <typename Sink, typename T, int Options>
void AbslStringify(Sink& sink, const Pose2<T, Options>& a_pose_b) {
  absl::Format(&sink, "translation: %v angle: %f",
               eigenmath::AbslStringified(a_pose_b.translation()),
               a_pose_b.angle());
}

template <typename T, int Options>
std::ostream& operator<<(std::ostream& os, const Pose2<T, Options>& a_pose_b) {
  os << absl::StrCat(a_pose_b);
  return os;
}

// SE(2) group multiplication operation for two 2D transformations
template <typename Scalar, int AOptions, int BOptions>
Pose2<Scalar> operator*(const Pose2<Scalar, AOptions>& a,
                        const Pose2<Scalar, BOptions>& b) {
  const Vector2<Scalar> translation =
      a.translation() + a.so2() * b.translation();
  const SO2<Scalar> rotation = a.so2() * b.so2();
  return Pose2<Scalar>{translation, rotation};
}

// SE(2) group multiplication operation for a Pose2 and a SO2 rotation,
// where the rotation is treated like a Pose2 with no translation.
template <typename Scalar, int AOptions, int BOptions>
Pose2<Scalar> operator*(const Pose2<Scalar, AOptions>& a,
                        const SO2<Scalar, BOptions>& b) {
  return Pose2<Scalar>{a.translation(), a.so2() * b};
}

// SE(2) group multiplication operation for a SO2 rotation and a Pose2
// where the rotation is treated like a Pose2 with no translation.
template <typename Scalar, int AOptions, int BOptions>
Pose2<Scalar> operator*(const SO2<Scalar, AOptions>& a,
                        const Pose2<Scalar, BOptions>& b) {
  const Vector2<Scalar> translation = a * b.translation();
  const SO2<Scalar> rotation = a * b.so2();
  return Pose2<Scalar>{translation, rotation};
}

// Transforms a 2D point with a 2D transformation
template <typename Scalar, typename Derived, int Options>
constexpr Vector2<Scalar> operator*(const Pose2<Scalar, Options>& pose,
                                    const Eigen::MatrixBase<Derived>& point) {
  return pose.translation() + pose.so2() * point;
}

using Pose2d = Pose2<double>;
using Pose2f = Pose2<float>;

}  // namespace eigenmath

#endif  // EIGENMATH_EIGENMATH_POSE2_H_
