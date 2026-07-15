from opendbc.car.ford.carcontroller import apply_brake_gas_interlock
from opendbc.car.ford.values import CarControllerParams


def test_brake_request_never_sends_positive_propulsion():
  assert apply_brake_gas_interlock(0.2, brake_request=True) == CarControllerParams.INACTIVE_GAS
  assert apply_brake_gas_interlock(0.2, brake_request=False) == 0.2
