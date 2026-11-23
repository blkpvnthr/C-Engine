import pytest

from cengine.dynamic_universe.config import UniverseRotationConfig


def test_defaults_match_spec():
    c = UniverseRotationConfig()
    assert c.active_universe_size == 30
    assert c.baseline_target_slots == 20
    assert c.opportunity_target_slots == 10
    assert c.refresh_interval_seconds == 60.0
    assert c.opportunity_refresh_seconds == 300.0
    assert c.baseline_max_revisit_seconds == 300.0
    assert c.candidate_min_residence_seconds == 180.0
    assert c.protect_open_positions is True
    assert c.protect_pending_orders is True


def test_from_env_overrides():
    env = {
        "CENGINE_UNIVERSE_ACTIVE_SIZE": "25",
        "CENGINE_UNIVERSE_BASELINE_SLOTS": "15",
        "CENGINE_UNIVERSE_OPPORTUNITY_SLOTS": "10",
        "CENGINE_UNIVERSE_MIN_RESIDENCE_S": "90",
    }
    c = UniverseRotationConfig.from_env(getenv=lambda k, d=None: env.get(k, d))
    assert c.active_universe_size == 25
    assert c.baseline_target_slots == 15
    assert c.candidate_min_residence_seconds == 90.0


def test_validate_rejects_bad_values():
    with pytest.raises(ValueError):
        UniverseRotationConfig(active_universe_size=0).validate()
    with pytest.raises(ValueError):
        UniverseRotationConfig(candidate_min_residence_seconds=-1).validate()
