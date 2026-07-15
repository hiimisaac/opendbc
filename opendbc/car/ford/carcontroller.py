import importlib
import math
import numpy as np
from opendbc.can import CANPacker
from opendbc.car import ACCELERATION_DUE_TO_GRAVITY, Bus, DT_CTRL, apply_hysteresis, structs
from opendbc.car.ford import fordcan
from opendbc.car.ford.lateral_path import FORD_PATH_UNWIND_ANGLE_DEADZONE_DEG, driver_steering_opposes_command, lateral_path_command
from opendbc.car.ford.values import CarControllerParams, FordFlags, CAR
from opendbc.car.interfaces import CarControllerBase, V_CRUISE_MAX
from opendbc.car.vehicle_model import VehicleModel


def _load_messaging(import_module=importlib.import_module):
  """Load messaging from sunnypilot first, with upstream as a fallback."""
  for module_name in ("cereal.messaging", "openpilot.cereal.messaging"):
    try:
      return import_module(module_name)
    except ImportError:
      pass
  return None


def lmc2_mode(lat_active: bool) -> int:
  return 2 if lat_active else 0


def lmc2_precision(cooperative_control: bool) -> int:
  return 0 if cooperative_control else 1


messaging = _load_messaging()

LongCtrlState = structs.CarControl.Actuators.LongControlState
VisualAlert = structs.CarControl.HUDControl.VisualAlert

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
    self.path_angle_last = 0.0
    self.path_offset_last = 0.0
    self.anti_overshoot_curvature_last = 0
    self.k_meas_filt = 0.0
    self.c0_undertrack_correction = 0.0
    self.c2_latched = False
    self.c2_recovery_frames = 0
    self.unwind_curvature = 0.0
    self.desired_curvature_last = 0.0
    self.curvature_rate_last = 0.0
    self.driver_handoff = False
    self.model = None
    self.model_frame = 0
    self.sm = None
    if messaging is not None and CP.flags & FordFlags.CANFD:
      try:
        self.sm = messaging.SubMaster(["modelV2"])
      except Exception:
        self.sm = None

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

    if self.sm is not None:
      self.sm.update(0)
      if self.sm.updated["modelV2"]:
        self.model = self.sm["modelV2"]
        self.model_frame = self.frame

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

    if (self.frame % CarControllerParams.STEER_STEP) == 0:
      desired_curvature = 0.0

      if CC.latActive:
        desired_curvature = actuators.curvature

        # Bronco and some other cars consistently overshoot curvature requests.
        # Apply the same input shaping before either Ford lateral command path.
        if self.CP.carFingerprint in (CAR.FORD_BRONCO_SPORT_MK1, CAR.FORD_F_150_MK14):
          self.anti_overshoot_curvature_last = anti_overshoot(desired_curvature, self.anti_overshoot_curvature_last, CS.out.vEgoRaw)
          desired_curvature = self.anti_overshoot_curvature_last

      if self.CP.flags & FordFlags.CANFD:
        current_curvature = -CS.out.yawRate / max(CS.out.vEgoRaw, 0.1)
        model = self.model if (self.frame - self.model_frame) * DT_CTRL < 0.5 else None
        angle_error_deg_raw = actuators.steeringAngleDeg - CS.out.steeringAngleDeg
        driver_override = CC.latActive and driver_steering_opposes_command(CS.out.steeringPressed, CS.out.steeringTorque,
                                                                           angle_error_deg_raw)
        cooperative_control = driver_override or self.driver_handoff
        # During an opposing override, synchronize the path to the steering
        # angle the driver is holding. Unlike yaw/v, this remains usable at low
        # speed and follows the driver's input without waiting for vehicle yaw.
        wheel_curvature = -self.VM.calc_curvature(math.radians(CS.out.steeringAngleDeg), CS.out.vEgoRaw, 0.0)
        path_curvature = current_curvature
        if driver_override:
          path_curvature = wheel_curvature

        # The normal lat-test controller remains curvature/model driven. Only
        # its large-turn flush may use the existing upstream steering-angle
        # target to actively return a wheel that is lagging the requested exit.
        angle_error_deg = angle_error_deg_raw
        angle_error_deg = math.copysign(max(abs(angle_error_deg) - FORD_PATH_UNWIND_ANGLE_DEADZONE_DEG, 0.0),
                                        angle_error_deg)
        angle_error_curvature = -self.VM.calc_curvature(math.radians(angle_error_deg), CS.out.vEgoRaw, 0.0)
        cmd = lateral_path_command(model, desired_curvature, path_curvature, CS.out.vEgoRaw,
                                   self.k_meas_filt, CC.latActive, driver_override,
                                   c2_last=self.apply_curvature_last,
                                   c0_undertrack_correction=self.c0_undertrack_correction,
                                   path_angle_last=self.path_angle_last,
                                   path_offset_last=self.path_offset_last,
                                   driver_handoff=self.driver_handoff and not driver_override,
                                   angle_error_curvature=angle_error_curvature,
                                   wheel_curvature=wheel_curvature,
                                   c2_latched_last=self.c2_latched,
                                   c2_recovery_frames_last=self.c2_recovery_frames,
                                   unwind_curvature_last=self.unwind_curvature,
                                   desired_curvature_last=self.desired_curvature_last,
                                   curvature_rate_last=self.curvature_rate_last)
        self.k_meas_filt = cmd.k_meas_filt
        self.c0_undertrack_correction = cmd.c0_undertrack_correction
        self.c2_latched = cmd.c2_latched
        self.c2_recovery_frames = cmd.c2_recovery_frames
        self.unwind_curvature = cmd.unwind_curvature
        self.desired_curvature_last = desired_curvature
        self.curvature_rate_last = cmd.curvature_rate
        apply_curvature = cmd.curvature
        curvature_rate = cmd.curvature_rate
        path_angle = cmd.path_angle
        path_offset = cmd.path_offset
        self.path_angle_last = path_angle
        self.path_offset_last = path_offset
        if driver_override:
          self.driver_handoff = True
        elif self.driver_handoff and cmd.handoff_complete:
          self.driver_handoff = False
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
    new_actuators.accel = self.accel
    new_actuators.gas = self.gas

    self.frame += 1
    return new_actuators, can_sends
