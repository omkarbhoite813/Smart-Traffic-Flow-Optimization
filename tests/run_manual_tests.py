from app import VehicleCounts, compute_signal_plan

# Test 1
lane0 = VehicleCounts(car=0, truck=0, motorcycle=0, bus=0, bicycle=0, ambulance=1, firetruck=0)
lane1 = VehicleCounts(car=10, truck=0, motorcycle=0, bus=0, bicycle=0, ambulance=0, firetruck=0)
plan = compute_signal_plan([lane0, lane1], cycle=60.0, min_green=5.0, max_green=60.0)
print('Plan greens:', plan.greens)
assert plan.greens[0] >= plan.greens[1], 'Emergency lane should have >= green time'

# Test 2
lanes = [VehicleCounts() for _ in range(4)]
plan2 = compute_signal_plan(lanes, cycle=60.0)
print('Plan2 greens:', plan2.greens)
assert len(plan2.greens) == 4
assert all(g >= 5.0 for g in plan2.greens)
print('Manual tests passed')
