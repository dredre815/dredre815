import importlib.util
import sys
from pathlib import Path

path = Path(__file__).parents[1] / "src/run_pilot.py"
spec = importlib.util.spec_from_file_location("pilot", path)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)
e = m.Event("X", "X", "2020-01-01T00:00:00+00:00", 1, 1.2, 20, 100, 110, 108, 1000, 2000, .1, .2, .1, .08, -.018, 1, -1)
clean = m.scenario(e, "clean")
attack = m.scenario(e, "splice")
assert clean["decision_snapshot_consistent"] is True
assert attack["decision_snapshot_consistent"] is False
assert attack["naive_ttl_pass"] is True
assert attack["naive_delivery_time_skew_5s_pass"] is True
assert m.visible(attack["event"]) == m.visible(clean["event"])
assert m.visible(attack["sentiment"]) == m.visible(clean["sentiment"])
assert m.visible(attack["market"]) != m.visible(clean["market"])
assert attack["market"]["retrieved_at"] == clean["market"]["retrieved_at"]
assert attack["market"]["_provenance"]["valid_to"] == attack["event"]["_provenance"]["valid_from"]
