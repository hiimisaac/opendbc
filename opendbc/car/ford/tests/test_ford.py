import random
import unittest
from types import SimpleNamespace

from opendbc.can.parser import CANParser
from hypothesis import settings, given, strategies as st

from opendbc.can import CANPacker
from opendbc.car import Bus, structs
from opendbc.car.ford.carcontroller import CarController
from opendbc.car.ford.interface import CarInterface
from opendbc.car.ford.radar_interface import RadarInterface
from opendbc.car.structs import CarParams
from opendbc.car.fw_versions import build_fw_dict
from opendbc.car.ford.values import CAR, DBC, FW_QUERY_CONFIG, FW_PATTERN, FordFlags, RADAR, CarControllerParams, get_platform_codes
from opendbc.car.ford.fingerprints import FW_VERSIONS
from opendbc.testing import parameterized

Ecu = CarParams.Ecu


ECU_ADDRESSES = {
  Ecu.eps: 0x730,          # Power Steering Control Module (PSCM)
  Ecu.abs: 0x760,          # Anti-Lock Brake System (ABS)
  Ecu.fwdRadar: 0x764,     # Cruise Control Module (CCM)
  Ecu.fwdCamera: 0x706,    # Image Processing Module A (IPMA)
  Ecu.engine: 0x7E0,       # Powertrain Control Module (PCM)
  Ecu.shiftByWire: 0x732,  # Gear Shift Module (GSM)
  Ecu.debug: 0x7D0,        # Accessory Protocol Interface Module (APIM)
}


ECU_PART_NUMBER = {
  Ecu.eps: [
    b"14D003",
  ],
  Ecu.abs: [
    b"2D053",
  ],
  Ecu.fwdRadar: [
    b"14D049",
  ],
  Ecu.fwdCamera: [
    b"14F397",  # Ford Q3
    b"14H102",  # Ford Q4
  ],
}


class TestFordFW(unittest.TestCase):
  def test_fw_query_config(self):
    for (ecu, addr, subaddr) in FW_QUERY_CONFIG.extra_ecus:
      assert ecu in ECU_ADDRESSES, "Unknown ECU"
      assert addr == ECU_ADDRESSES[ecu], "ECU address mismatch"
      assert subaddr is None, "Unexpected ECU subaddress"

  @parameterized("car_model, fw_versions", FW_VERSIONS.items())
  def test_fw_versions(self, car_model, fw_versions):
    for (ecu, addr, subaddr), fws in fw_versions.items():
      assert ecu in ECU_PART_NUMBER, "Unexpected ECU"
      assert addr == ECU_ADDRESSES[ecu], "ECU address mismatch"
      assert subaddr is None, "Unexpected ECU subaddress"

      for fw in fws:
        assert len(fw) == 24, "Expected ECU response to be 24 bytes"

        match = FW_PATTERN.match(fw)
        assert match is not None, f"Unable to parse FW: {fw!r}"
        if match:
          part_number = match.group("part_number")
          assert part_number in ECU_PART_NUMBER[ecu], f"Unexpected part number for {fw!r}"

        codes = get_platform_codes([fw])
        assert 1 == len(codes), f"Unable to parse FW: {fw!r}"

  @settings(max_examples=100)
  @given(data=st.data())
  def test_platform_codes_fuzzy_fw(self, data):
    """Ensure function doesn't raise an exception"""
    fw_strategy = st.lists(st.binary())
    fws = data.draw(fw_strategy)
    get_platform_codes(fws)

  def test_platform_codes_spot_check(self):
    # Asserts basic platform code parsing behavior for a few cases
    results = get_platform_codes([
      b"JX6A-14C204-BPL\x00\x00\x00\x00\x00\x00\x00\x00\x00",
      b"NZ6T-14F397-AC\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",
      b"PJ6T-14H102-ABJ\x00\x00\x00\x00\x00\x00\x00\x00\x00",
      b"LB5A-14C204-EAC\x00\x00\x00\x00\x00\x00\x00\x00\x00",
    ])
    assert results == {(b"X6A", b"J"), (b"Z6T", b"N"), (b"J6T", b"P"), (b"B5A", b"L")}

  def test_fuzzy_match(self):
    for platform, fw_by_addr in FW_VERSIONS.items():
      # Ensure there's no overlaps in platform codes
      for _ in range(20):
        car_fw = []
        for ecu, fw_versions in fw_by_addr.items():
          ecu_name, addr, sub_addr = ecu
          fw = random.choice(fw_versions)
          car_fw.append(CarParams.CarFw(ecu=ecu_name, fwVersion=fw, address=addr,
                                        subAddress=0 if sub_addr is None else sub_addr))

        CP = CarParams(carFw=car_fw)
        matches = FW_QUERY_CONFIG.match_fw_to_car_fuzzy(build_fw_dict(CP.carFw), CP.carVin, FW_VERSIONS)
        assert matches == {platform}

  def test_match_fw_fuzzy(self):
    offline_fw = {
      (Ecu.eps, 0x730, None): [
        b"L1MC-14D003-AJ\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",
        b"L1MC-14D003-AL\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",
      ],
      (Ecu.abs, 0x760, None): [
        b"L1MC-2D053-BA\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",
        b"L1MC-2D053-BD\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",
      ],
      (Ecu.fwdRadar, 0x764, None): [
        b"LB5T-14D049-AB\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",
        b"LB5T-14D049-AD\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",
      ],
      # We consider all model year hints for ECU, even with different platform codes
      (Ecu.fwdCamera, 0x706, None): [
        b"LB5T-14F397-AD\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",
        b"NC5T-14F397-AF\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",
      ],
    }
    expected_fingerprint = CAR.FORD_EXPLORER_MK6

    # ensure that we fuzzy match on all non-exact FW with changed revisions
    live_fw = {
      (0x730, None): {b"L1MC-14D003-XX\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"},
      (0x760, None): {b"L1MC-2D053-XX\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"},
      (0x764, None): {b"LB5T-14D049-XX\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"},
      (0x706, None): {b"LB5T-14F397-XX\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"},
    }
    candidates = FW_QUERY_CONFIG.match_fw_to_car_fuzzy(live_fw, '', {expected_fingerprint: offline_fw})
    assert candidates == {expected_fingerprint}

    # model year hint in between the range should match
    live_fw[(0x706, None)] = {b"MB5T-14F397-XX\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"}
    candidates = FW_QUERY_CONFIG.match_fw_to_car_fuzzy(live_fw, '', {expected_fingerprint: offline_fw,})
    assert candidates == {expected_fingerprint}

    # unseen model year hint should not match
    live_fw[(0x760, None)] = {b"M1MC-2D053-XX\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"}
    candidates = FW_QUERY_CONFIG.match_fw_to_car_fuzzy(live_fw, '', {expected_fingerprint: offline_fw})
    assert len(candidates) == 0, "Should not match new model year hint"


class TestFordRadar(unittest.TestCase):
  def test_canfd_platforms_have_radar(self):
    for platform in CAR.with_flags(FordFlags.CANFD):
      CP = CarInterface.get_non_essential_params(platform)
      assert not CP.radarUnavailable
      self.assertAlmostEqual(CP.radarDelay, 0.1)

  @staticmethod
  def _canfd_radar_frames(packer: CANPacker, scan_index: int, bus: int, detections=None):
    frames = []
    if detections is None:
      detections = [
        {"msg_idx": 1, "slot_idx": 1, "range": 40.0 + scan_index, "range_rate": -1.0, "azimuth": 0.0},
      ]

    for msg_idx in range(1, 23):
      slots = 3 if msg_idx == 22 else 6
      values = {
        f"CAN_SCAN_INDEX_2LSB_{msg_idx:02d}_{slot_idx:02d}": scan_index
        for slot_idx in range(1, slots + 1)
      }

      for det in detections:
        if det["msg_idx"] != msg_idx:
          continue
        slot_idx = det["slot_idx"]
        values |= {
          f"CAN_DET_VALID_LEVEL_{msg_idx:02d}_{slot_idx:02d}": 1,
          f"CAN_DET_RANGE_{msg_idx:02d}_{slot_idx:02d}": det["range"],
          f"CAN_DET_RANGE_RATE_{msg_idx:02d}_{slot_idx:02d}": det["range_rate"],
          f"CAN_DET_AZIMUTH_{msg_idx:02d}_{slot_idx:02d}": det["azimuth"],
          f"CAN_DET_HOST_VEH_CLUTTER_{msg_idx:02d}_{slot_idx:02d}": int(det.get("host_veh_clutter", False)),
        }

      frames.append(packer.make_can_msg(f"MRR_Detection_{msg_idx:03d}", bus, values))
    return frames

  def test_canfd_radar_interface_updates(self):
    CP = CarInterface.get_non_essential_params(CAR.FORD_F_150_LIGHTNING_MK1)
    RI = RadarInterface(CP)
    packer = CANPacker(RADAR.DELPHI_MRR_CANFD)

    updates = []
    for frame, scan_index in enumerate((0, 1, 2, 3, 0, 1, 2, 3), start=1):
      frames = self._canfd_radar_frames(packer, scan_index, RI.rcp.bus)
      updates.append(RI.update([(frame * 50_000_000, frames)]))

    assert all(radar_data is not None for radar_data in updates)
    assert len(updates[3].points) > 0
    assert [len(radar_data.points) for radar_data in updates[4:]] == [len(updates[3].points)] * 4

  def test_canfd_radar_clusters_nearby_detections(self):
    CP = CarInterface.get_non_essential_params(CAR.FORD_F_150_LIGHTNING_MK1)
    RI = RadarInterface(CP)
    packer = CANPacker(RADAR.DELPHI_MRR_CANFD)
    detections = [
      {"msg_idx": 1, "slot_idx": 1, "range": 40.0, "range_rate": -1.0, "azimuth": 0.0},
      {"msg_idx": 1, "slot_idx": 2, "range": 41.0, "range_rate": -1.1, "azimuth": 0.01},
      {"msg_idx": 2, "slot_idx": 1, "range": 39.5, "range_rate": -0.9, "azimuth": -0.01},
    ]

    updates = []
    for frame, scan_index in enumerate((0, 1, 2, 3), start=1):
      frames = self._canfd_radar_frames(packer, scan_index, RI.rcp.bus, detections=detections)
      updates.append(RI.update([(frame * 50_000_000, frames)]))

    assert len(updates[-1].points) == 1

  def test_canfd_radar_reuses_track_after_short_miss(self):
    CP = CarInterface.get_non_essential_params(CAR.FORD_F_150_LIGHTNING_MK1)
    RI = RadarInterface(CP)
    packer = CANPacker(RADAR.DELPHI_MRR_CANFD)
    detections = [
      {"msg_idx": 1, "slot_idx": 1, "range": 40.0, "range_rate": -1.0, "azimuth": 0.0},
      {"msg_idx": 1, "slot_idx": 2, "range": 40.5, "range_rate": -1.1, "azimuth": 0.01},
    ]

    frame = 1
    track_id = None
    for cycle_detections in (detections, [], detections):
      radar_data = None
      for scan_index in (0, 1, 2, 3):
        frames = self._canfd_radar_frames(packer, scan_index, RI.rcp.bus, detections=cycle_detections)
        radar_data = RI.update([(frame * 50_000_000, frames)])
        frame += 1

      if cycle_detections:
        assert len(radar_data.points) == 1
        if track_id is None:
          track_id = radar_data.points[0].trackId
        else:
          assert radar_data.points[0].trackId == track_id

  def test_canfd_radar_rejects_host_vehicle_clutter(self):
    CP = CarInterface.get_non_essential_params(CAR.FORD_F_150_LIGHTNING_MK1)
    RI = RadarInterface(CP)
    packer = CANPacker(RADAR.DELPHI_MRR_CANFD)
    detections = [
      {"msg_idx": 1, "slot_idx": 1, "range": 8.0, "range_rate": -1.0, "azimuth": 0.0, "host_veh_clutter": True},
    ]

    radar_data = None
    for frame, scan_index in enumerate((0, 1, 2, 3), start=1):
      frames = self._canfd_radar_frames(packer, scan_index, RI.rcp.bus, detections=detections)
      radar_data = RI.update([(frame * 50_000_000, frames)])

    assert len(radar_data.points) == 0


class TestFordCanfdLateral(unittest.TestCase):
  def test_canfd_lateral_motion_control_keeps_desired_curvature_in_path(self):
    CP = CarInterface.get_non_essential_params(CAR.FORD_F_150_LIGHTNING_MK1)
    CC = structs.CarControl()
    CC.latActive = True
    CC.actuators.curvature = 0.002
    CC.hudControl.leadDistanceBars = 1

    CS_out = structs.CarState()
    CS_out.vEgo = 12.0
    CS_out.vEgoRaw = 12.0
    CS_out.cruiseState.available = True
    CS = SimpleNamespace(
      out=CS_out.as_reader(),
      buttons_stock_values={},
      acc_tja_status_stock_values={"Tja_D_Stat": 0},
      lkas_status_stock_values={},
    )

    controller = CarController(DBC[CP.carFingerprint], CP)
    controller.frame = CarControllerParams.STEER_STEP
    controller.main_on_last = True
    controller.lkas_enabled_last = True
    controller.lead_distance_bars_last = 1

    _, can_sends = controller.update(CC.as_reader(), CS, 0)

    parser = CANParser(DBC[CP.carFingerprint][Bus.pt], [("LateralMotionControl2", 20)], 0)
    parser.update([0, can_sends])

    values = parser.vl["LateralMotionControl2"]
    self.assertGreater(abs(values["LatCtlPathOffst_L_Actl"]), 0.0)
    self.assertGreater(abs(values["LatCtlPath_An_Actl"]), 0.0)
    self.assertEqual(values["LatCtlCurv_No_Actl"], 0.0)
