import math

import numpy as np
from opendbc.can import CANPacker
from opendbc.car import ACCELERATION_DUE_TO_GRAVITY, Bus, DT_CTRL, apply_hysteresis, structs
from opendbc.car.ford import fordcan
from opendbc.car.ford.lateral_path import driver_steering_opposes_command, SteeringAngleProjector
from opendbc.car.ford.lateral_path_projector import ProjectedLatControlPath
from opendbc.car.ford.values import CarControllerParams, FordFlags, CAR
from opendbc.car.interfaces import CarControllerBase, V_CRUISE_MAX
from opendbc.car.vehicle_model import VehicleModel


def lmc2_mode(lat_active: bool) -> int:
  return 2 if lat_active else 0


def lmc2_precision(cooperative_control: bool) -> int:
  return 0 if cooperative_control else 1


LongCtrlState = structs.CarControl.Actuators.LongControlState
VisualAlert = structs.CarControl.HUDControl.VisualAlert

def ford_curvature_from_steering_angle(VM, steering_angle_deg: float, v_ego: float) -> float:
  """Convert steering-wheel angle to Ford's opposite-sign curvature."""
  return -float(VM.calc_curvature(math.radians(steering_angle_deg), v_ego, 0.0))


def anti_overshoot(apply_curvature, apply_curvature_last, v_ego):
  diff = 0.1
  tau = 5  # 5s smooths over the overshoot
  dt = DT_CTRL * CarControllerParams.STEER_STEP
  alpha = 1 - np.exp(-dt / tau)

  lataccel = apply_curvature * (v_ego ** 2)
  last_lataccel = apply_curvature_last * (v_ego ** 2)
  last_lataccel = apply_hysteresis(lataccel, last_lataccel, diff)
  last_lataccel = alpha * lataccel + (1 - alpha) * last_lataccel

  output_curvature = last_lataccel / (max(v_ego, 1) ** 2)

  return float(np.interp(v_ego, [5, 10], [apply_curvature, output_curvature]))


def apply_creep_compensation(accel: float, v_ego: float) -> float:
  creep_accel = np.interp(v_ego, [1., 3.], [0.6, 0.])
  creep_accel = np.interp(accel, [0., 0.2], [creep_accel, 0.])
  accel -= creep_accel
  return float(accel)


class CarController(CarControllerBase):
  def __init__(self, dbc_names, CP):
    super().__init__(dbc_names, CP)
    self.packer = CANPacker(dbc_names[Bus.pt])
    self.CAN = fordcan.CanBus(CP)
    self.VM = VehicleModel(CP)

    self.apply_curvature_last = 0
    self.path_offset_last = 0.0
    self.path_angle_last = 0.0
    self.curvature_rate_last = 0.0
    self.path_valid_last = False
    self.anti_overshoot_curvature_last = 0
    self.lateral_path_controller = ProjectedLatControlPath()
    self.steering_angle_projector = SteeringAngleProjector()

    self.accel = 0.0
    self.gas = 0.0
    self.brake_request = False
    self.main_on_last = False
    self.lkas_enabled_last = False
    self.steer_alert_last = False
    self.lead_distance_bars_last = None
    self.distance_bar_frame = 0

  def update(self, CC, CS, now_nanos):
    can_sends = []

    actuators = CC.actuators
    hud_control = CC.hudControl

    main_on = CS.out.cruiseState.available
    steer_alert = hud_control.visualAlert in (VisualAlert.steerRequired, VisualAlert.ldw)
    fcw_alert = hud_control.visualAlert == VisualAlert.fcw

    ### acc buttons ###
    if CC.cruiseControl.cancel:
      can_sends.append(fordcan.create_button_msg(self.packer, self.CAN.camera, CS.buttons_stock_values, cancel=True))
      can_sends.append(fordcan.create_button_msg(self.packer, self.CAN.main, CS.buttons_stock_values, cancel=True))
    elif CC.cruiseControl.resume and (self.frame % CarControllerParams.BUTTONS_STEP) == 0:
      can_sends.append(fordcan.create_button_msg(self.packer, self.CAN.camera, CS.buttons_stock_values, resume=True))
      can_sends.append(fordcan.create_button_msg(self.packer, self.CAN.main, CS.buttons_stock_values, resume=True))
    # if stock lane centering isn't off, send a button press to toggle it off
    # the stock system checks for steering pressed, and eventually disengages cruise control
    elif CS.acc_tja_status_stock_values["Tja_D_Stat"] != 0 and (self.frame % CarControllerParams.ACC_UI_STEP) == 0:
      can_sends.append(fordcan.create_button_msg(self.packer, self.CAN.camera, CS.buttons_stock_values, tja_toggle=True))

    ### lateral control ###
    # send steer msg at 20Hz
    apply_curvature = 0.0
    path_angle = 0.0
    path_offset = 0.0
    curvature_rate = 0.0
    ramp_type = 3
    driver_override = False
    cooperative_control = False

    if (self.frame % CarControllerParams.STEER_STEP) == 0:
      desired_curvature = 0.0

      if CC.latActive:
        desired_curvature = (actuators.lateralPath.curvature if self.CP.flags & FordFlags.CANFD else
                             actuators.curvature)

        # Bronco and some other cars consistently overshoot curvature requests.
        # Apply the same input shaping before either Ford lateral command path.
        if self.CP.carFingerprint in (CAR.FORD_BRONCO_SPORT_MK1, CAR.FORD_F_150_MK14):
          self.anti_overshoot_curvature_last = anti_overshoot(desired_curvature, self.anti_overshoot_curvature_last, CS.out.vEgoRaw)
          desired_curvature = self.anti_overshoot_curvature_last

      if self.CP.flags & FordFlags.CANFD:
        angle_error_deg_raw = actuators.steeringAngleDeg - CS.out.steeringAngleDeg
        actual_angle_deg = CS.out.steeringAngleDeg
        projected_angle_deg = self.steering_angle_projector.update(actual_angle_deg)
        driver_override = driver_steering_opposes_command(
          CC.latActive and CS.out.steeringPressed,
          CS.out.steeringTorque,
          angle_error_deg_raw,
        )
        cooperative_control = driver_override
        measured_curvature = ford_curvature_from_steering_angle(self.VM, actual_angle_deg, CS.out.vEgoRaw)
        projected_wheel_curvature = ford_curvature_from_steering_angle(self.VM, projected_angle_deg, CS.out.vEgoRaw)
        desired_angle_curvature = ford_curvature_from_steering_angle(
          self.VM, actuators.steeringAngleDeg, CS.out.vEgoRaw,
        )
        path_target = actuators.lateralPath
        if desired_curvature != path_target.curvature:
          path_target = path_target.as_builder()
          path_target.curvature = desired_curvature
        cmd = self.lateral_path_controller.update(
          path_target, measured_curvature, CS.out.vEgoRaw,
          CC.latActive, driver_override,
          projected_measured_curvature=projected_wheel_curvature,
          desired_angle_curvature=desired_angle_curvature,
        )
        apply_curvature = cmd.curvature
        curvature_rate = cmd.curvature_rate
        path_angle = cmd.path_angle
        path_offset = cmd.path_offset
        self.path_valid_last = cmd.valid
      elif CC.latActive:
        current_curvature = -CS.out.yawRate / max(CS.out.vEgoRaw, 0.1)
        # Preserve upstream's curvature error and ISO lateral jerk limits for
        # non-CAN FD Ford platforms. CAN FD uses the bounded LMC2 polynomial.
        if CS.out.vEgoRaw > 9:
          desired_curvature = float(np.clip(desired_curvature, current_curvature - CarControllerParams.CURVATURE_ERROR,
                                            current_curvature + CarControllerParams.CURVATURE_ERROR))
        apply_curvature = CarControllerParams.CURVATURE_LIMITS.apply_limits(
          desired_curvature, self.apply_curvature_last, CS.out.vEgoRaw,
          0., CC.latActive, CarControllerParams.STEER_STEP,
        )

      self.apply_curvature_last = apply_curvature
      self.path_offset_last = path_offset
      self.path_angle_last = path_angle
      self.curvature_rate_last = curvature_rate

      if self.CP.flags & FordFlags.CANFD:
        mode = lmc2_mode(CC.latActive)
        precision = lmc2_precision(cooperative_control)
        counter = (self.frame // CarControllerParams.STEER_STEP) % 0x10
        can_sends.append(fordcan.create_lat_ctl2_msg(
          self.packer, self.CAN, mode, ramp_type, precision, -path_offset, -path_angle,
          -apply_curvature, -curvature_rate, counter
        ))
      else:
        can_sends.append(fordcan.create_lat_ctl_msg(self.packer, self.CAN, CC.latActive, 0., 0., -self.apply_curvature_last, 0.))

    # send lka msg at 33Hz
    if (self.frame % CarControllerParams.LKA_STEP) == 0:
      can_sends.append(fordcan.create_lka_msg(self.packer, self.CAN))

    ### longitudinal control ###
    # send acc msg at 50Hz
    if self.CP.openpilotLongitudinalControl and (self.frame % CarControllerParams.ACC_CONTROL_STEP) == 0:
      accel = actuators.accel
      gas = accel

      if CC.longActive:
        # Compensate for engine creep at low speed.
        # Either the ABS does not account for engine creep, or the correction is very slow
        # TODO: verify this applies to EV/hybrid
        accel = apply_creep_compensation(accel, CS.out.vEgo)

        # The stock system has been seen rate limiting the brake accel to 5 m/s^3,
        # however even 3.5 m/s^3 causes some overshoot with a step response.
        accel = max(accel, self.accel - (3.5 * CarControllerParams.ACC_CONTROL_STEP * DT_CTRL))

      accel = float(np.clip(accel, CarControllerParams.ACCEL_MIN, CarControllerParams.ACCEL_MAX))
      gas = float(np.clip(gas, CarControllerParams.ACCEL_MIN, CarControllerParams.ACCEL_MAX))

      # Both gas and accel are in m/s^2, accel is used solely for braking
      if not CC.longActive or gas < CarControllerParams.MIN_GAS:
        gas = CarControllerParams.INACTIVE_GAS

      # PCM applies pitch compensation to gas/accel, but we need to compensate for the brake/pre-charge bits
      accel_due_to_pitch = 0.0
      if len(CC.orientationNED) == 3:
        accel_due_to_pitch = math.sin(CC.orientationNED[1]) * ACCELERATION_DUE_TO_GRAVITY

      accel_pitch_compensated = accel + accel_due_to_pitch
      if accel_pitch_compensated > 0.3 or not CC.longActive:
        self.brake_request = False
      elif accel_pitch_compensated < 0.0:
        self.brake_request = True

      stopping = CC.actuators.longControlState == LongCtrlState.stopping
      # TODO: look into using the actuators packet to send the desired speed
      can_sends.append(fordcan.create_acc_msg(self.packer, self.CAN, CC.longActive, gas, accel, stopping, self.brake_request, v_ego_kph=V_CRUISE_MAX))

      self.accel = accel
      self.gas = gas

    ### ui ###
    send_ui = (self.main_on_last != main_on) or (self.lkas_enabled_last != CC.latActive) or (self.steer_alert_last != steer_alert)
    # send lkas ui msg at 1Hz or if ui state changes
    if (self.frame % CarControllerParams.LKAS_UI_STEP) == 0 or send_ui:
      can_sends.append(fordcan.create_lkas_ui_msg(self.packer, self.CAN, main_on, CC.latActive, steer_alert, hud_control, CS.lkas_status_stock_values))

    # send acc ui msg at 5Hz or if ui state changes
    if hud_control.leadDistanceBars != self.lead_distance_bars_last:
      send_ui = True
      self.distance_bar_frame = self.frame

    if (self.frame % CarControllerParams.ACC_UI_STEP) == 0 or send_ui:
      show_distance_bars = self.frame - self.distance_bar_frame < 400
      can_sends.append(fordcan.create_acc_ui_msg(self.packer, self.CAN, self.CP, main_on, CC.latActive,
                                                 fcw_alert, CS.out.cruiseState.standstill, show_distance_bars,
                                                 hud_control, CS.acc_tja_status_stock_values))

    self.main_on_last = main_on
    self.lkas_enabled_last = CC.latActive
    self.steer_alert_last = steer_alert
    self.lead_distance_bars_last = hud_control.leadDistanceBars

    new_actuators = actuators.as_builder()
    new_actuators.curvature = self.apply_curvature_last
    new_actuators.lateralPath.valid = self.path_valid_last
    new_actuators.lateralPath.pathOffset = self.path_offset_last
    new_actuators.lateralPath.pathAngle = self.path_angle_last
    new_actuators.lateralPath.curvature = self.apply_curvature_last
    new_actuators.lateralPath.curvatureRate = self.curvature_rate_last
    new_actuators.accel = self.accel
    new_actuators.gas = self.gas

    self.frame += 1
    return new_actuators, can_sends
