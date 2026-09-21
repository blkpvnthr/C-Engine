from datetime import date

import h5py

from market_data.market_data_store import create_daily_file


def test_daily_hdf5_has_explicit_provider_and_schema(tmp_path):
    path = create_daily_file(tmp_path, date(2026, 1, 2), provider="deterministic-test")
    with h5py.File(path, "r") as handle:
        assert handle.attrs["provider"] == "deterministic-test"
        assert handle.attrs["schema_version"] == 2
        assert "equities" in handle
