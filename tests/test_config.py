"""Tests for configuration."""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from inflation_disaster.config import settings


class TestConfig:
    def test_bin_edges_length(self):
        """9 edges for 8 bins."""
        assert len(settings.bin_edges) == 9

    def test_bin_edges_sorted(self):
        edges = list(settings.bin_edges)
        for i in range(1, len(edges)):
            assert edges[i] > edges[i - 1]

    def test_bin_midpoints_length(self):
        assert len(settings.bin_midpoints) == 8

    def test_strikes_count(self):
        """From -2 to 6 in 0.5 steps = 17 strikes."""
        assert len(settings.strikes) == 17

    def test_strikes_endpoints(self):
        assert settings.strikes[0] == -2.0
        assert settings.strikes[-1] == 6.0

    def test_strikes_spacing(self):
        diffs = np.diff(settings.strikes)
        np.testing.assert_allclose(diffs, 0.5, atol=1e-10)

    def test_paper_constants(self):
        assert settings.target_inflation == 2.0
        assert settings.rra == 3.0
        assert settings.risk_adj_high == 0.66
        assert settings.risk_adj_low == 0.96
        assert settings.us_p_mr == 0.50
        assert settings.ez_p_mr == 0.47
