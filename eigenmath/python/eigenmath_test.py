import unittest

import numpy as np

import eigenmath


class TestBasic(unittest.TestCase):

  np_atol = 1.0e-12

  def test_constructor_vector3d(self):
    expected_translation_vector = [0.1, 0.2, 3.0]
    pose = eigenmath.Pose3d(expected_translation_vector)
    np.testing.assert_allclose(pose.translation,
                               expected_translation_vector,
                               atol=self.np_atol)

  def test_constructor_quaterniond(self):
    expected_quaternion_wxyz = [
        0.5 * np.sqrt(2.0), 0.0, 0.5 * np.sqrt(2.0), 0.0
    ]
    pose = eigenmath.Pose3d(expected_quaternion_wxyz)
    np.testing.assert_allclose(pose.quaternion_wxyz,
                               expected_quaternion_wxyz,
                               atol=self.np_atol)

  def test_constructor_matrix3d(self):
    expected_matrix3d = np.eye(3)
    expected_matrix3d[0, 0] = 0.0
    expected_matrix3d[0, 1] = -1.0
    expected_matrix3d[1, 0] = 1.0
    expected_matrix3d[1, 1] = 0.0
    pose = eigenmath.Pose3d(expected_matrix3d)
    np.testing.assert_allclose(pose.rotation_matrix,
                               expected_matrix3d,
                               atol=self.np_atol)

  def test_constructor_matrix4d(self):
    expected_translation_vector = [0.1, 0.2, 3.0]
    expected_matrix3d = np.eye(3)
    expected_matrix3d[0, 0] = 0.0
    expected_matrix3d[0, 1] = -1.0
    expected_matrix3d[1, 0] = 1.0
    expected_matrix3d[1, 1] = 0.0
    expected_matrix4d = np.eye(4)
    expected_matrix4d[:3, :3] = expected_matrix3d
    expected_matrix4d[:3, 3] = expected_translation_vector
    pose = eigenmath.Pose3d(expected_matrix4d)
    np.testing.assert_allclose(pose.rotation_matrix,
                               expected_matrix3d,
                               atol=self.np_atol)
    np.testing.assert_allclose(pose.translation,
                               expected_translation_vector,
                               atol=self.np_atol)

  def test_constructor_quaterniond_vector3d(self):
    expected_quaternion_wxyz = [
        0.5 * np.sqrt(2.0), 0.0, 0.5 * np.sqrt(2.0), 0.0
    ]
    expected_translation_vector = [0.1, 0.2, 3.0]
    pose = eigenmath.Pose3d(expected_quaternion_wxyz,
                            expected_translation_vector)
    np.testing.assert_allclose(pose.quaternion_wxyz,
                               expected_quaternion_wxyz,
                               atol=self.np_atol)
    np.testing.assert_allclose(pose.translation,
                               expected_translation_vector,
                               atol=self.np_atol)

  def test_constructor_matrix3d_vector3d(self):
    expected_translation_vector = [0.1, 0.2, 3.0]
    expected_matrix3d = np.eye(3)
    expected_matrix3d[0, 0] = 0.0
    expected_matrix3d[0, 1] = -1.0
    expected_matrix3d[1, 0] = 1.0
    expected_matrix3d[1, 1] = 0.0
    pose = eigenmath.Pose3d(expected_matrix3d, expected_translation_vector)
    np.testing.assert_allclose(pose.rotation_matrix,
                               expected_matrix3d,
                               atol=self.np_atol)
    np.testing.assert_allclose(pose.translation,
                               expected_translation_vector,
                               atol=self.np_atol)

  def test_multiplication_pose(self):
    expected_translation_a = [0.5, 0.0, 2.0]
    pose_a = eigenmath.Pose3d(expected_translation_a)
    expected_translation_b = [1.0, 2.0, 3.0]
    pose_b = eigenmath.Pose3d(expected_translation_b)
    expected_translation_final = [1.5, 2.0, 5.0]
    pose_b *= pose_a
    np.testing.assert_allclose(pose_a.translation,
                               expected_translation_a,
                               atol=self.np_atol)
    np.testing.assert_allclose(pose_b.translation,
                               expected_translation_final,
                               atol=self.np_atol)

  def test_multiplication_pose_pose(self):
    expected_translation_a = [0.5, 0.0, 2.0]
    pose_a = eigenmath.Pose3d(expected_translation_a)
    expected_translation_b = [1.0, 2.0, 3.0]
    pose_b = eigenmath.Pose3d(expected_translation_b)
    expected_translation_c = [1.5, 2.0, 5.0]
    pose_c = pose_a * pose_b
    np.testing.assert_allclose(pose_a.translation,
                               expected_translation_a,
                               atol=self.np_atol)
    np.testing.assert_allclose(pose_b.translation,
                               expected_translation_b,
                               atol=self.np_atol)
    np.testing.assert_allclose(pose_c.translation,
                               expected_translation_c,
                               atol=self.np_atol)

  def test_multiplication_point(self):
    expected_translation_a = [0.5, 0.0, 2.0]
    pose_a = eigenmath.Pose3d(expected_translation_a)
    point_b = [1.0, 2.0, 3.0]
    expected_translation_final = [1.5, 2.0, 5.0]
    point_c = pose_a * point_b
    np.testing.assert_allclose(point_c,
                               expected_translation_final,
                               atol=self.np_atol)

  def test_multiplication_point_list(self):
    expected_translation_a = [0.5, 0.0, 2.0]
    pose_a = eigenmath.Pose3d(expected_translation_a)
    points_b = np.transpose([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    expected_translation_final = [[1.5, 2.0, 5.0], [4.5, 5.0, 8.0]]
    points_c = pose_a * points_b
    np.testing.assert_allclose(points_c.T,
                               expected_translation_final,
                               atol=self.np_atol)

  def test_translation(self):
    expected_translation_vector = [0.1, 0.2, 3.0]
    pose = eigenmath.Pose3d()
    pose.translation = expected_translation_vector
    np.testing.assert_allclose(pose.translation,
                               expected_translation_vector,
                               atol=self.np_atol)

  def test_rotation_vector(self):
    expected_rotation_vector = [0.5, 1.0, 1.5]
    pose = eigenmath.Pose3d()
    pose.rotation_vector = expected_rotation_vector
    np.testing.assert_allclose(pose.rotation_vector,
                               expected_rotation_vector,
                               atol=self.np_atol)

  def test_quaternion_wxyz(self):
    expected_quaternion_wxyz = [
        0.5 * np.sqrt(2.0), 0.0, 0.5 * np.sqrt(2.0), 0.0
    ]
    pose = eigenmath.Pose3d()
    pose.quaternion_wxyz = expected_quaternion_wxyz
    np.testing.assert_allclose(pose.quaternion_wxyz,
                               expected_quaternion_wxyz,
                               atol=self.np_atol)

  def test_rotation_matrix(self):
    expected_matrix3d = np.eye(3)
    expected_matrix3d[0, 0] = 0.0
    expected_matrix3d[0, 1] = -1.0
    expected_matrix3d[1, 0] = 1.0
    expected_matrix3d[1, 1] = 0.0
    pose = eigenmath.Pose3d()
    pose.rotation_matrix = expected_matrix3d
    np.testing.assert_allclose(pose.rotation_matrix,
                               expected_matrix3d,
                               atol=self.np_atol)

  def test_axes(self):
    expected_matrix3d = np.eye(3)
    expected_matrix3d[0, 0] = 0.0
    expected_matrix3d[0, 1] = -1.0
    expected_matrix3d[1, 0] = 1.0
    expected_matrix3d[1, 1] = 0.0
    pose = eigenmath.Pose3d(expected_matrix3d)
    np.testing.assert_allclose(pose.x_axis,
                               expected_matrix3d[:, 0],
                               atol=self.np_atol)
    np.testing.assert_allclose(pose.y_axis,
                               expected_matrix3d[:, 1],
                               atol=self.np_atol)
    np.testing.assert_allclose(pose.z_axis,
                               expected_matrix3d[:, 2],
                               atol=self.np_atol)

  def test_inverse(self):
    pose = eigenmath.RotationVector([1, 0.2, -0.3]) * eigenmath.Pose3d(
        [0.5, 1.0, 2.0])
    pose_inv = pose.inverse()
    pose_identity = pose * pose_inv
    np.testing.assert_allclose(pose_identity.matrix,
                               np.eye(4),
                               atol=self.np_atol)

  def test_rotation_vector_builder(self):
    expected_rotation_vector = [0.5, 1.0, 1.5]
    pose = eigenmath.RotationVector(expected_rotation_vector)
    np.testing.assert_allclose(pose.rotation_vector,
                               expected_rotation_vector,
                               atol=self.np_atol)

  def test_rotation_x_builder(self):
    rotation_amount = 0.5
    expected_rotation_vector = [rotation_amount, 0.0, 0.0]
    pose = eigenmath.RotationX(rotation_amount)
    np.testing.assert_allclose(pose.rotation_vector,
                               expected_rotation_vector,
                               atol=self.np_atol)

  def test_rotation_y_builder(self):
    rotation_amount = 0.5
    expected_rotation_vector = [0.0, rotation_amount, 0.0]
    pose = eigenmath.RotationY(rotation_amount)
    np.testing.assert_allclose(pose.rotation_vector,
                               expected_rotation_vector,
                               atol=self.np_atol)

  def test_rotation_z_builder(self):
    rotation_amount = 0.5
    expected_rotation_vector = [0.0, 0.0, rotation_amount]
    pose = eigenmath.RotationZ(rotation_amount)
    np.testing.assert_allclose(pose.rotation_vector,
                               expected_rotation_vector,
                               atol=self.np_atol)


if __name__ == "__main__":
  unittest.main()
