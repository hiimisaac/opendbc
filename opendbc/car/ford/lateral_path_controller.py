from dataclasses import replace

from opendbc.car.ford.lateral_path import LateralPathCommand, lateral_path_command


class LateralPathController:
  """Own the persistent state of Ford's composed polynomial controller."""

  def __init__(self):
    self._curvature = 0.0
    self._curvature_rate = 0.0
    self._path_angle = 0.0
    self._path_offset = 0.0
    self._k_meas_filt = 0.0
    self._c2_latched = False
    self._c2_recovery_frames = 0
    self._unwind_curvature = 0.0
    self._driver_handoff = False

  @property
  def handoff_active(self) -> bool:
    return self._driver_handoff

  def update(self, model, desired_curvature: float, k_meas: float, v_ego: float,
             lat_active: bool, driver_override: bool, *,
             angle_error_curvature: float = 0.0,
             wheel_curvature: float = 0.0,
             projected_wheel_curvature: float | None = None,
             hold_command: bool = False) -> LateralPathCommand:
    cmd = lateral_path_command(
      model, desired_curvature, k_meas, v_ego,
      self._k_meas_filt, lat_active, driver_override,
      c2_last=self._curvature,
      path_angle_last=self._path_angle,
      path_offset_last=self._path_offset,
      driver_handoff=self._driver_handoff and not driver_override,
      angle_error_curvature=angle_error_curvature,
      wheel_curvature=wheel_curvature,
      projected_wheel_curvature=projected_wheel_curvature,
      c2_latched_last=self._c2_latched,
      c2_recovery_frames_last=self._c2_recovery_frames,
      unwind_curvature_last=self._unwind_curvature,
      curvature_rate_last=self._curvature_rate,
    )

    # Filter and latch state advance during the one-frame road-shock hold.
    # Coefficient and unwind state advance only for transmitted commands.
    self._k_meas_filt = cmd.k_meas_filt
    self._c2_latched = cmd.c2_latched
    self._c2_recovery_frames = cmd.c2_recovery_frames
    if hold_command:
      cmd = replace(cmd,
                    curvature=self._curvature,
                    curvature_rate=self._curvature_rate,
                    path_angle=self._path_angle,
                    path_offset=self._path_offset,
                    unwind_curvature=self._unwind_curvature)
    else:
      self._curvature = cmd.curvature
      self._curvature_rate = cmd.curvature_rate
      self._path_angle = cmd.path_angle
      self._path_offset = cmd.path_offset
      self._unwind_curvature = cmd.unwind_curvature

    if driver_override:
      self._driver_handoff = True
    elif self._driver_handoff and cmd.handoff_complete:
      self._driver_handoff = False

    return cmd
