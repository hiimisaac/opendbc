import math
from types import SimpleNamespace

from opendbc.car.ford.carcontroller import (
  _load_messaging,
  ford_curvature_from_steering_angle,
  hold_lateral_command,
  lmc2_mode,
  lmc2_precision,
)
from opendbc.car.ford.fordcan import lmc2_curvature_rate_for_can
from opendbc.car.ford.lateral_path_controller import (
  FORD_PATH_C0_CAN_CLIP,
  FORD_PATH_C1_CAN_CLIP,
  FORD_PATH_C2_CAN_CLIP,
  FORD_PATH_C3_CAN_CLIP,
  LateralPathController,
  model_curvature_rate,
)


def polynomial_model(curvature: float, curvature_rate: float = 0.0):
  distances = (0.0, 1.75, 2.5, 3.5, 5.0, 7.0, 12.0, 20.0)
  return SimpleNamespace(
    position=SimpleNamespace(
      x=list(distances),
      y=[0.5 * curvature * s**2 + curvature_rate * s**3 / 6.0 for s in distances],
    ),
    orientation=SimpleNamespace(
      z=[curvature * s + 0.5 * curvature_rate * s**2 for s in distances],
    ),
  )


def test_c3_never_adds_to_a_turn_when_wheel_is_already_past_model_target():
  controller = LateralPathController()

  command = controller.update(
    model=polynomial_model(0.01, 0.0004),
    desired_curvature=0.01,
    k_meas=0.02,
    v_ego=7.0,
    lat_active=True,
    driver_override=False,
    angle_error_curvature=-0.01,
    wheel_curvature=0.02,
    projected_wheel_curvature=0.02,
  )

  assert command.curvature_rate == 0.0


def test_large_turn_keeps_c2_out_until_target_and_wheel_settle():
  controller = LateralPathController()
  controller.update(
    polynomial_model(0.015), 0.015, 0.012, 7.0, True, False, angle_error_curvature=0.003, wheel_curvature=0.012, projected_wheel_curvature=0.012
  )

  for _ in range(50):
    command = controller.update(
      polynomial_model(0.004), 0.004, 0.004, 7.0, True, False, angle_error_curvature=0.0, wheel_curvature=0.004, projected_wheel_curvature=0.004
    )
  assert command.curvature == 0.0

  for _ in range(9):
    command = controller.update(
      polynomial_model(0.001), 0.001, 0.001, 7.0, True, False, angle_error_curvature=0.0, wheel_curvature=0.001, projected_wheel_curvature=0.001
    )
  assert command.curvature == 0.0

  command = controller.update(
    polynomial_model(0.001), 0.001, 0.001, 7.0, True, False, angle_error_curvature=0.0, wheel_curvature=0.001, projected_wheel_curvature=0.001
  )
  assert command.curvature == 0.0002


def test_undertracking_feedback_is_bounded_to_twenty_percent_of_model_target():
  open_controller = LateralPathController()
  closed_controller = LateralPathController()
  for _ in range(10):
    open_loop = open_controller.update(
      polynomial_model(0.01),
      0.01,
      0.0,
      7.0,
      True,
      False,
      angle_error_curvature=0.01,
      wheel_curvature=0.0,
    )
    closed_loop = closed_controller.update(
      polynomial_model(0.01),
      0.01,
      0.0,
      7.0,
      True,
      False,
      angle_error_curvature=0.01,
      wheel_curvature=0.0,
      projected_wheel_curvature=0.0,
    )

  max_c0_feedback = 0.5 * (0.2 * 0.01) * 7.0**2
  assert 0.0 < closed_loop.path_offset - open_loop.path_offset <= max_c0_feedback


def test_still_wound_wheel_gets_active_unwind_without_discrete_turn_state():
  controller = LateralPathController()
  for _ in range(3):
    controller.update(polynomial_model(0.01), 0.01, 0.01, 7.0, True, False, angle_error_curvature=0.0, wheel_curvature=0.01, projected_wheel_curvature=0.01)

  command = controller.update(
    polynomial_model(0.0),
    0.0,
    0.008,
    7.0,
    True,
    False,
    angle_error_curvature=-0.008,
    wheel_curvature=0.008,
    projected_wheel_curvature=0.008,
  )

  assert command.path_angle < 0.0
  assert command.path_offset < 0.0


def test_driver_input_holds_growth_for_both_pending_and_helping_torque():
  assert hold_lateral_command(steering_pressed=True, driver_override=False, pending=False)
  assert hold_lateral_command(steering_pressed=True, driver_override=True, pending=False) is False
  assert hold_lateral_command(steering_pressed=False, driver_override=False, pending=True)
  assert hold_lateral_command(steering_pressed=False, driver_override=False, pending=False) is False


def test_matching_gentle_path_uses_quiet_c2_channel():
  controller = LateralPathController()
  for _ in range(20):
    command = controller.update(
      polynomial_model(0.003),
      0.003,
      0.003,
      15.0,
      True,
      False,
      angle_error_curvature=0.0,
      wheel_curvature=0.003,
      projected_wheel_curvature=0.003,
    )

  assert command.curvature == 0.003
  assert command.path_angle == 0.0
  assert command.path_offset == 0.0
  assert command.curvature_rate == 0.0


def test_c0_c1_keep_model_turn_authority_missing_from_action():
  controller = LateralPathController()
  for _ in range(20):
    command = controller.update(
      polynomial_model(0.01),
      0.0034,
      0.01,
      7.0,
      True,
      False,
      angle_error_curvature=0.0,
      wheel_curvature=0.01,
      projected_wheel_curvature=0.01,
    )

  assert command.curvature + command.path_angle / 7.0 >= 0.0095
  assert command.path_offset > 0.15


def test_c0_c1_keep_action_authority_when_near_model_geometry_is_weaker():
  controller = LateralPathController()
  for _ in range(4):
    command = controller.update(
      polynomial_model(0.012),
      0.02,
      0.012,
      7.0,
      True,
      False,
      angle_error_curvature=0.008,
      wheel_curvature=0.012,
      projected_wheel_curvature=0.012,
    )

  assert command.path_angle > 0.13
  assert command.path_offset > 0.45


def test_coherent_farther_geometry_advances_a_building_moving_turn():
  command = LateralPathController().update(
    polynomial_model(0.004, 0.001),
    0.004,
    0.004,
    7.0,
    True,
    False,
    angle_error_curvature=0.0,
    wheel_curvature=0.004,
    projected_wheel_curvature=0.004,
  )

  assert command.path_angle > 0.03
  assert command.path_offset > 0.1


def test_c3_assists_a_coherent_curvature_reversal():
  command = LateralPathController().update(
    polynomial_model(-0.008, -0.0004),
    -0.0034,
    0.004,
    7.0,
    True,
    False,
    angle_error_curvature=-0.012,
    wheel_curvature=0.004,
    projected_wheel_curvature=0.004,
  )

  assert command.curvature_rate < 0.0


def test_hold_returns_exact_last_transmitted_polynomial():
  controller = LateralPathController()
  first = controller.update(polynomial_model(0.008), 0.008, 0.0, 7.0, True, False)
  held = controller.update(polynomial_model(0.02), 0.02, 0.0, 7.0, True, False, hold_command=True)
  resumed = controller.update(polynomial_model(0.02), 0.02, 0.0, 7.0, True, False)

  assert held == first
  assert resumed != held


def test_driver_override_tracks_wheel_then_handoff_is_bounded():
  controller = LateralPathController()
  override = controller.update(
    polynomial_model(0.02),
    0.02,
    0.012,
    7.0,
    True,
    True,
    wheel_curvature=0.012,
    projected_wheel_curvature=0.012,
  )
  handoff = controller.update(
    polynomial_model(-0.01),
    -0.01,
    0.012,
    7.0,
    True,
    False,
    angle_error_curvature=-0.022,
    wheel_curvature=0.012,
    projected_wheel_curvature=0.012,
  )

  assert override.curvature == 0.0
  assert override.curvature_rate == 0.0
  assert math.isclose(override.path_angle, 0.012 * 7.0)
  assert math.isclose(override.path_offset, 0.5 * 0.012 * 7.0**2)
  assert abs(handoff.path_angle - override.path_angle) <= 0.01 * 7.0
  assert abs(handoff.path_offset - override.path_offset) <= 0.5 * 0.01 * 7.0**2


def test_relatch_cannot_wind_past_wheel_while_upstream_is_unwinding():
  controller = LateralPathController()
  controller.update(
    polynomial_model(0.08),
    0.0,
    0.06,
    7.0,
    True,
    True,
    wheel_curvature=0.06,
    projected_wheel_curvature=0.06,
  )
  command = controller.update(
    polynomial_model(0.08),
    0.0,
    0.06,
    7.0,
    True,
    False,
    angle_error_curvature=-0.06,
    wheel_curvature=0.06,
    projected_wheel_curvature=0.06,
  )

  assert command.path_angle <= 0.06 * 7.0
  assert command.path_offset <= 0.5 * 0.06 * 7.0**2


def test_inactive_and_nonfinite_inputs_are_safe_and_bounded():
  controller = LateralPathController()
  inactive = controller.update(polynomial_model(0.02), 0.02, 0.0, 7.0, False, False)
  command = controller.update(
    polynomial_model(10.0, 1.0),
    float("nan"),
    float("inf"),
    7.0,
    True,
    False,
    angle_error_curvature=float("nan"),
    wheel_curvature=float("inf"),
    projected_wheel_curvature=float("nan"),
  )

  assert inactive.curvature == inactive.curvature_rate == 0.0
  assert inactive.path_angle == inactive.path_offset == 0.0
  assert FORD_PATH_C0_CAN_CLIP[0] <= command.path_offset <= FORD_PATH_C0_CAN_CLIP[1]
  assert FORD_PATH_C1_CAN_CLIP[0] <= command.path_angle <= FORD_PATH_C1_CAN_CLIP[1]
  assert FORD_PATH_C2_CAN_CLIP[0] <= command.curvature <= FORD_PATH_C2_CAN_CLIP[1]
  assert FORD_PATH_C3_CAN_CLIP[0] <= command.curvature_rate <= FORD_PATH_C3_CAN_CLIP[1]


def test_model_curvature_rate_uses_spatial_distance():
  distances = (0.0, 5.0, 10.0)
  curvature_rate = 0.0006
  model = SimpleNamespace(
    position=SimpleNamespace(x=[0.0, 3.0, 6.0], y=[0.0, 4.0, 8.0]),
    orientation=SimpleNamespace(
      z=[0.01 * s + 0.5 * curvature_rate * s**2 for s in distances],
    ),
  )

  assert math.isclose(model_curvature_rate(model, 10.0), curvature_rate, abs_tol=1e-12)


def test_lmc2_helpers_keep_full_rate_range_and_cooperative_mode():
  assert lmc2_curvature_rate_for_can(0.0008) == 0.0008
  assert lmc2_curvature_rate_for_can(-0.0008) == -0.0008
  assert lmc2_curvature_rate_for_can(0.002) == 0.001023
  assert lmc2_curvature_rate_for_can(-0.002) == -0.001024
  assert lmc2_curvature_rate_for_can(float("nan")) == 0.0
  assert lmc2_mode(True) == 2
  assert lmc2_mode(False) == 0
  assert lmc2_precision(True) == 0
  assert lmc2_precision(False) == 1


def test_model_subscription_supports_both_embedding_namespaces():
  sentinel = object()
  imports = []

  def standard_import(name):
    imports.append(name)
    if name == "cereal.messaging":
      return sentinel
    raise ImportError(name)

  assert _load_messaging(standard_import) is sentinel
  assert imports == ["cereal.messaging"]

  imports.clear()

  def upstream_import(name):
    imports.append(name)
    if name == "openpilot.cereal.messaging":
      return sentinel
    raise ImportError(name)

  assert _load_messaging(upstream_import) is sentinel
  assert imports == ["cereal.messaging", "openpilot.cereal.messaging"]


def test_ford_steering_angle_conversion_uses_curvature_sign():
  class FakeVehicleModel:
    @staticmethod
    def calc_curvature(angle_rad, _speed, _roll):
      return angle_rad / 10.0

  assert ford_curvature_from_steering_angle(FakeVehicleModel(), 10.0, 5.0) < 0.0
  assert ford_curvature_from_steering_angle(FakeVehicleModel(), -10.0, 5.0) > 0.0
