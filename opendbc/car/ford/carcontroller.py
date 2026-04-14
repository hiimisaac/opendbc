import math
import numpy as np
from numpy import clip
from opendbc.can import CANPacker
from opendbc.car import ACCELERATION_DUE_TO_GRAVITY, Bus, DT_CTRL, apply_hysteresis, structs
from opendbc.car.lateral import ISO_LATERAL_ACCEL, apply_std_steer_angle_limits
from opendbc.car.ford import fordcan
from opendbc.car.ford.values import CarControllerParams, FordFlags, CAR
from opendbc.car.interfaces import CarControllerBase, V_CRUISE_MAX

LongCtrlState = structs.CarControl.Actuators.LongControlState
VisualAlert = structs.CarControl.HUDControl.VisualAlert

# CAN FD limits:
# Limit to average banked road since safety doesn't have the roll
AVERAGE_ROAD_ROLL = 0.06  # ~3.4 degrees, 6% superelevation. higher actual roll raises lateral acceleration
MAX_LATERAL_ACCEL = ISO_LATERAL_ACCEL - (ACCELERATION_DUE_TO_GRAVITY * AVERAGE_ROAD_ROLL)  # ~2.4 m/s^2

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


def apply_ford_curvature_limits(apply_curvature, apply_curvature_last, current_curvature, v_ego_raw, steering_angle, lat_active, CP):
  # No blending at low speed due to lack of torque wind-up and inaccurate current curvature
  if v_ego_raw > 9:
    apply_curvature = np.clip(apply_curvature, current_curvature - CarControllerParams.CURVATURE_ERROR,
                              current_curvature + CarControllerParams.CURVATURE_ERROR)

  # Curvature rate limit after driver torque limit
  apply_curvature = apply_std_steer_angle_limits(apply_curvature, apply_curvature_last, v_ego_raw, steering_angle, lat_active, CarControllerParams.ANGLE_LIMITS)

  # Ford Q4/CAN FD has more torque available compared to Q3/CAN so we limit it based on lateral acceleration.
  # Safety is not aware of the road roll so we subtract a conservative amount at all times
  if CP.flags & FordFlags.CANFD:
    # Limit curvature to conservative max lateral acceleration
    curvature_accel_limit = MAX_LATERAL_ACCEL / (max(v_ego_raw, 1) ** 2)
    apply_curvature = float(np.clip(apply_curvature, -curvature_accel_limit, curvature_accel_limit))

  return apply_curvature


def first_order_filter(val: float, val_last: float, tau: float) -> float:
  dt = DT_CTRL * CarControllerParams.STEER_STEP
  alpha = 1 - np.exp(-dt / tau) if tau > 0 else 1.0
  return float(alpha * val + (1.0 - alpha) * val_last)


def suppress_curvature_sign_flip(curvature: float, curvature_last: float, deadband: float) -> float:
  if abs(curvature) < deadband:
    return 0.0
  if curvature * curvature_last < 0 and abs(curvature) < 2.0 * deadband:
    return 0.0
  return curvature


def get_ford_canfd_mode(lat_active: bool, capability_status: int) -> int:
  if not lat_active:
    return 0
  if capability_status >= 2:
    return 2
  if capability_status == 1:
    return 1
  return 0


def get_ford_canfd_c0_lookahead(v_ego: float, d_look: float) -> float:
  c0_lookahead = np.interp(v_ego, [8.0, 14.0, 25.0], [8.0, 6.0, 4.0])
  return float(min(d_look, c0_lookahead))


def get_ford_canfd_path_offset_trim(path_offset: float, path_angle: float, curvature_target: float,
                                    path_offset_trim_last: float, v_ego: float, lat_active: bool) -> float:
  trim_target = 0.0
  if lat_active:
    path_angle_limit = float(np.interp(v_ego, *CarControllerParams.C0_TRIM_PATH_ANGLE_MAX))
    curvature_limit = float(np.interp(v_ego, *CarControllerParams.C0_TRIM_CURVATURE_MAX))
    if abs(path_angle) < path_angle_limit and abs(curvature_target) < curvature_limit:
      trim_gain = float(np.interp(v_ego, *CarControllerParams.C0_TRIM_GAIN))
      trim_max = float(np.interp(v_ego, *CarControllerParams.C0_TRIM_MAX))
      trim_target = float(np.clip(path_offset * trim_gain, -trim_max, trim_max))

  return first_order_filter(trim_target, path_offset_trim_last, CarControllerParams.C0_TRIM_TAU)


def get_ford_canfd_c1_lookahead(v_ego: float, d_look: float, curvature_last: float, curvature_target: float, limit_status: int) -> float:
  lookahead = d_look
  if abs(curvature_target) < abs(curvature_last) or curvature_target * curvature_last < 0:
    lookahead *= float(np.interp(v_ego, *CarControllerParams.C1_EXIT_LOOKAHEAD_FACTOR))
  lookahead *= float(np.interp(float(limit_status), *CarControllerParams.C1_LIMIT_LOOKAHEAD_FACTOR))
  return float(np.clip(lookahead, CarControllerParams.C1_LOOKAHEAD_MIN, d_look))


def get_ford_curvature_filter_tau(curvature: float, curvature_last: float) -> float:
  if curvature * curvature_last < 0:
    return CarControllerParams.C2_CROSSOVER_TAU
  if abs(curvature) > abs(curvature_last):
    return CarControllerParams.C2_WINDUP_TAU
  return CarControllerParams.C2_UNWIND_TAU


def shape_ford_canfd_curvature(curvature: float, curvature_last: float, limit_status: int, capability_status: int) -> float:
  # When the PSCM reports limited/near-limit, back off new wind-up but preserve fast unwind.
  if abs(curvature) > abs(curvature_last):
    curvature *= float(np.interp(float(limit_status), *CarControllerParams.C2_LIMIT_FACTOR))
    if capability_status == 1:
      curvature *= CarControllerParams.C2_LIMITED_MODE_FACTOR
  return curvature


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

    self.apply_curvature_last = 0
    self.anti_overshoot_curvature_last = 0
    self.path_angle_last = 0.0
    self.path_offset_last = 0.0
    self.path_offset_trim_last = 0.0
    self.curvature_target_last = 0.0

    try:
      import cereal.messaging as messaging
      self.sm = messaging.SubMaster(['modelV2'])
      self.has_model = True
    except ImportError:
      self.sm = None
      self.has_model = False
    self.model = None

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

    if self.has_model and self.sm is not None:
      self.sm.update(0)
      if self.sm.updated['modelV2']:
        self.model = self.sm['modelV2']

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
    path_offset_trim = 0.0
    path_offset_trim_updated = False
    curvature_target = 0.0
    curvature_rate = 0.0
    ramp_type = 0

    if (self.frame % CarControllerParams.STEER_STEP) == 0:
      canfd_capability_status = getattr(CS, "lat_ctl_capability_status", 2)
      canfd_limit_status = getattr(CS, "lat_ctl_limit_status", 0)
      canfd_mode = get_ford_canfd_mode(CC.latActive, canfd_capability_status) if self.CP.flags & FordFlags.CANFD else 0
      canfd_lat_active = canfd_mode != 0

      if CC.latActive:
        v_ego = CS.out.vEgoRaw
        desired_curvature = actuators.curvature

        if self.CP.flags & FordFlags.CANFD:
          current_curvature = -CS.out.yawRate / max(v_ego, 0.1)
          curvature_deadband = float(np.interp(v_ego, *CarControllerParams.C2_SIGN_FLIP_DEADBAND))
          curvature_target = desired_curvature if canfd_lat_active else 0.0
          curvature_target = suppress_curvature_sign_flip(curvature_target, self.apply_curvature_last, curvature_deadband)
          curvature_target = shape_ford_canfd_curvature(curvature_target, self.curvature_target_last, canfd_limit_status, canfd_capability_status)
          curvature_tau = get_ford_curvature_filter_tau(curvature_target, self.curvature_target_last)
          curvature_target = first_order_filter(curvature_target, self.curvature_target_last, curvature_tau)
          apply_curvature = apply_ford_curvature_limits(curvature_target, self.apply_curvature_last, current_curvature,
                                                        v_ego, 0., canfd_lat_active, self.CP)

          if self.model is not None and len(self.model.position.x) > 0 and len(self.model.position.y) > 0 and len(self.model.orientation.z) > 0:
            x_pts = np.array(self.model.position.x)
            d_look = max(v_ego * 1.0, 7.0)
            d_c1 = get_ford_canfd_c1_lookahead(v_ego, d_look, self.curvature_target_last, curvature_target, canfd_limit_status)
            path_angle = float(np.interp(d_c1, x_pts, np.array(self.model.orientation.z)))

            d_c0 = get_ford_canfd_c0_lookahead(v_ego, d_look)
            path_offset = float(np.interp(d_c0, x_pts, np.array(self.model.position.y)))
            path_offset_trim = get_ford_canfd_path_offset_trim(path_offset, path_angle, curvature_target,
                                                               self.path_offset_trim_last, v_ego, canfd_lat_active)
            path_offset_trim_updated = True
            path_offset = float(path_offset + path_offset_trim)
            path_offset = apply_hysteresis(path_offset, self.path_offset_last, CarControllerParams.C0_HYSTERESIS)
            path_offset = apply_std_steer_angle_limits(
              path_offset, self.path_offset_last, v_ego, 0., canfd_lat_active, CarControllerParams.C0_RATE_LIMITS)

            path_angle = apply_std_steer_angle_limits(
              path_angle, self.path_angle_last, v_ego, 0., canfd_lat_active, CarControllerParams.C1_RATE_LIMITS)

            ramp_type = 3

            path_offset_limit = float(np.interp(v_ego, *CarControllerParams.C0_MAX))
            path_offset = float(clip(path_offset, -path_offset_limit, path_offset_limit))
            path_offset = float(clip(path_offset, -4.61, 4.60))
            path_angle = float(clip(path_angle, -0.475, 0.497))
        else:
          # Non-CAN FD: curvature-only control (unchanged from upstream)
          # Bronco and some other cars consistently overshoot curv requests
          # Apply some deadzone + smoothing convergence to avoid oscillations
          if self.CP.carFingerprint in (CAR.FORD_BRONCO_SPORT_MK1, CAR.FORD_F_150_MK14):
            self.anti_overshoot_curvature_last = anti_overshoot(desired_curvature, self.anti_overshoot_curvature_last, CS.out.vEgoRaw)
            apply_curvature = self.anti_overshoot_curvature_last
          else:
            apply_curvature = desired_curvature

          # apply rate limits, curvature error limit, and clip to signal range
          current_curvature = -CS.out.yawRate / max(CS.out.vEgoRaw, 0.1)
          apply_curvature = apply_ford_curvature_limits(apply_curvature, self.apply_curvature_last, current_curvature,
                                                        CS.out.vEgoRaw, 0., CC.latActive, self.CP)

      self.path_angle_last = path_angle
      self.path_offset_last = path_offset
      if path_offset_trim_updated:
        self.path_offset_trim_last = path_offset_trim
      else:
        self.path_offset_trim_last = first_order_filter(0.0, self.path_offset_trim_last, CarControllerParams.C0_TRIM_TAU)
      self.curvature_target_last = curvature_target
      self.apply_curvature_last = apply_curvature

      if self.CP.flags & FordFlags.CANFD:
        counter = (self.frame // CarControllerParams.STEER_STEP) % 0x10
        can_sends.append(fordcan.create_lat_ctl2_msg(
          self.packer, self.CAN, canfd_mode, ramp_type, 1, -path_offset, -path_angle,
          -apply_curvature, -curvature_rate, counter
        ))
      else:
        can_sends.append(fordcan.create_lat_ctl_msg(
          self.packer, self.CAN, CC.latActive, 0., 0., -self.apply_curvature_last, 0.
        ))

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
    new_actuators.accel = self.accel
    new_actuators.gas = self.gas

    self.frame += 1
    return new_actuators, can_sends
