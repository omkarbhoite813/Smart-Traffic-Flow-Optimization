import pytest

from app import VehicleCounts, compute_signal_plan


def test_emergency_priority_over_regular():
    # Lane 0 has an ambulance, lane 1 has many cars
    lane0 = VehicleCounts(car=0, truck=0, motorcycle=0, bus=0, bicycle=0, ambulance=1, firetruck=0)
    lane1 = VehicleCounts(car=10, truck=0, motorcycle=0, bus=0, bicycle=0, ambulance=0, firetruck=0)

    plan = compute_signal_plan([lane0, lane1], cycle=60.0, min_green=5.0, max_green=60.0)

    # Expect lane 0 (with ambulance) to get >= green time than lane 1 due to emergency weighting
    assert plan.greens[0] >= plan.greens[1]


def test_all_zero_counts():
    lanes = [VehicleCounts() for _ in range(4)]
    plan = compute_signal_plan(lanes, cycle=60.0)
    # With no vehicles, greens should be distributed (min_green applied)
    assert len(plan.greens) == 4
    assert all(g >= 5.0 for g in plan.greens)
