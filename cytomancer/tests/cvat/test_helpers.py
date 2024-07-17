import pytest

import numpy as np

from cytomancer.cvat.helpers import _parse_field_selector
from cytomancer.experiment import Axes


class TestParseFieldSelector:
    def test_accepts_hyphen_separated_strings(self):
        selector = "field-1"
        axis, field_value = _parse_field_selector(selector)
        assert axis == "field"
        assert field_value == '1'

    def test_accepts_axes_names_for_field_names(self):
        for axis in Axes:
            selector = f"{axis}-1"
            axis_, _ = _parse_field_selector(selector)
            assert axis_ == axis

    def test_rejects_non_axes_names_for_field_names(self):
        with pytest.raises(ValueError):
            _parse_field_selector("foo-1")

    def test_field_values_containing_hyphens_are_supported(self):
        selector = "field-1-2"
        _, field_value = _parse_field_selector(selector)
        assert field_value == '1-2'

    def test_time_value_is_parsed_as_ns_datetime(self):
        selector = "time-1"
        _, field_value = _parse_field_selector(selector)
        assert field_value == np.datetime64(1, 'ns')

    def test_field_values_containing_colons_are_interpreted_as_arrays(self):
        selector = "field-1:2"
        _, field_value = _parse_field_selector(selector)
        assert (field_value == np.array(["1", "2"])).all()
