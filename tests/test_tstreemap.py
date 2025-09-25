import tempfile
import time
from pathlib import Path

import pytest

from laser_generic.newutils import TimingStats
from laser_generic.tstreemap import _convert_to_d3_hierarchy
from laser_generic.tstreemap import generate_d3_treemap_html


class TestTreemapGeneration:
    def test_basic_d3_hierarchy_conversion(self):
        """Test conversion of TimingStats data to D3 hierarchy format."""
        timer = TimingStats()

        with timer.start("parent"):
            time.sleep(0.01)
            with timer.start("child1"):
                time.sleep(0.005)
            with timer.start("child2"):
                time.sleep(0.005)

        timer.freeze()

        hierarchy = _convert_to_d3_hierarchy(timer._timing_data, scale_factor=1_000_000)

        # Should have parent node with children
        assert hierarchy["name"] == "parent"
        assert hierarchy["value"] > 0
        assert hierarchy["call_count"] == 1
        assert "children" in hierarchy
        assert len(hierarchy["children"]) == 2

        # Children should be in execution order
        child_names = [child["name"] for child in hierarchy["children"]]
        assert child_names == ["child1", "child2"]

        # Each child should have proper timing data
        for child in hierarchy["children"]:
            assert child["value"] > 0
            assert child["call_count"] == 1
            assert "execution_order" in child

    def test_multiple_root_nodes_d3_conversion(self):
        """Test D3 conversion with multiple root-level timers."""
        timer = TimingStats()

        with timer.start("root1"):
            time.sleep(0.005)

        with timer.start("root2"):
            time.sleep(0.005)

        timer.freeze()

        hierarchy = _convert_to_d3_hierarchy(timer._timing_data, scale_factor=1_000_000)

        # Should create wrapper node
        assert hierarchy["name"] == "Total Execution"
        assert "children" in hierarchy
        assert len(hierarchy["children"]) == 2

        child_names = [child["name"] for child in hierarchy["children"]]
        assert child_names == ["root1", "root2"]

    def test_empty_timing_data_d3_conversion(self):
        """Test D3 conversion with only global timing data."""
        timer = TimingStats()
        timer.freeze()

        hierarchy = _convert_to_d3_hierarchy(timer._timing_data, scale_factor=1_000_000)

        # Should create basic wrapper
        assert hierarchy["name"] == "Total Execution"
        assert hierarchy["value"] > 0
        assert hierarchy["call_count"] == 1
        assert hierarchy["children"] == []

    def test_scale_factor_conversion(self):
        """Test that scale factor correctly converts nanoseconds."""
        timer = TimingStats()

        with timer.start("test"):
            time.sleep(0.1)  # 100ms

        timer.freeze()

        # Test millisecond conversion
        hierarchy_ms = _convert_to_d3_hierarchy(timer._timing_data, scale_factor=1_000_000)
        assert 90 < hierarchy_ms["value"] < 110  # Around 100ms

        # Test second conversion
        hierarchy_s = _convert_to_d3_hierarchy(timer._timing_data, scale_factor=1_000_000_000)
        assert 0.09 < hierarchy_s["value"] < 0.11  # Around 0.1s

    def test_html_file_generation(self):
        """Test that HTML file is generated correctly."""
        timer = TimingStats()

        with timer.start("test_task"):
            time.sleep(0.01)
            with timer.start("subtask"):
                time.sleep(0.005)

        timer.freeze()

        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "test_treemap.html"

            generate_d3_treemap_html(timer, str(output_file), title="Test Treemap", scale="ms", width=800, height=600)

            # File should exist
            assert output_file.exists()

            # File should contain expected content
            content = output_file.read_text(encoding="utf-8")
            assert "Test Treemap" in content
            assert "d3js.org/d3.v7.min.js" in content
            assert "test_task" in content
            assert "subtask" in content
            assert 'width="800"' in content
            assert 'height="600"' in content
            assert "ms" in content

    def test_html_generation_unfrozen_timer_raises_error(self):
        """Test that generating HTML from unfrozen timer raises error."""
        timer = TimingStats()

        with timer.start("test"):
            time.sleep(0.001)

        # Don't freeze the timer
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "test.html"

            with pytest.raises(RuntimeError, match="must be frozen"):
                generate_d3_treemap_html(timer, str(output_file))

    def test_invalid_scale_raises_error(self):
        """Test that invalid scale parameter raises error."""
        timer = TimingStats()

        with timer.start("test"):
            time.sleep(0.001)

        timer.freeze()

        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "test.html"

            with pytest.raises(ValueError, match="Invalid scale"):
                generate_d3_treemap_html(timer, str(output_file), scale="invalid")

    def test_different_scales_in_html(self):
        """Test HTML generation with different time scales."""
        timer = TimingStats()

        with timer.start("test"):
            time.sleep(0.05)  # 50ms

        timer.freeze()

        # scales_to_test = ["ns", "microseconds", "ms", "s"]
        scales_to_test = ["ns", "µs", "ms", "s"]

        for scale in scales_to_test:
            with tempfile.TemporaryDirectory() as temp_dir:
                output_file = Path(temp_dir) / f"test_{scale}.html"

                generate_d3_treemap_html(timer, str(output_file), scale=scale)

                content = output_file.read_text(encoding="utf-8")
                # Check that scale unit is present in the JavaScript
                # if scale == "microseconds":
                #     # The μ character might be encoded differently, so check for the pattern
                #     assert 'scaleUnit = "' in content and 's"' in content
                # else:
                #     assert f'"{scale}"' in content
                assert f'"{scale}"' in content

    def test_execution_order_preserved_in_d3_data(self):
        """Test that execution order is preserved in D3 hierarchy data."""
        timer = TimingStats()

        # Create timers in non-alphabetical order
        with timer.start("zebra"):
            time.sleep(0.001)

        with timer.start("alpha"):
            time.sleep(0.001)

        with timer.start("beta"):
            time.sleep(0.001)

        timer.freeze()

        hierarchy = _convert_to_d3_hierarchy(timer._timing_data, scale_factor=1_000_000)

        # Should have wrapper with children in execution order
        assert hierarchy["name"] == "Total Execution"
        child_names = [child["name"] for child in hierarchy["children"]]
        assert child_names == ["zebra", "alpha", "beta"]  # Execution order, not alphabetical

        # Verify execution order values
        for i, child in enumerate(hierarchy["children"], 1):
            assert child["execution_order"] == i

    def test_nested_execution_order_in_d3_data(self):
        """Test execution order preservation in nested hierarchy."""
        timer = TimingStats()

        with timer.start("parent"):
            time.sleep(0.001)
            with timer.start("child_z"):
                time.sleep(0.001)
            with timer.start("child_a"):
                time.sleep(0.001)
            with timer.start("child_m"):
                time.sleep(0.001)

        timer.freeze()

        hierarchy = _convert_to_d3_hierarchy(timer._timing_data, scale_factor=1_000_000)

        # Check nested children order
        children = hierarchy["children"]
        child_names = [child["name"] for child in children]
        assert child_names == ["child_z", "child_a", "child_m"]

    def test_self_time_calculation_in_d3_data(self):
        """Test that self_time is correctly calculated in D3 data."""
        timer = TimingStats()

        with timer.start("parent"):
            time.sleep(0.01)  # 10ms parent self-time
            with timer.start("child"):
                time.sleep(0.01)  # 10ms child time

        timer.freeze()

        hierarchy = _convert_to_d3_hierarchy(timer._timing_data, scale_factor=1_000_000)

        # Parent should have both value and self_value
        assert hierarchy["value"] > hierarchy["self_value"]
        assert hierarchy["self_value"] > 0

        # Child should have self_value approximately equal to value
        child = hierarchy["children"][0]
        assert abs(child["value"] - child["self_value"]) < 2  # Allow 2ms tolerance

    def test_html_contains_interactive_features(self):
        """Test that generated HTML contains interactive D3 features."""
        timer = TimingStats()

        with timer.start("interactive_test"):
            time.sleep(0.01)
            with timer.start("sub_feature"):
                time.sleep(0.005)

        timer.freeze()

        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "interactive_test.html"

            generate_d3_treemap_html(timer, str(output_file))

            content = output_file.read_text(encoding="utf-8")

            # Should contain interactive JavaScript features
            assert "mouseover" in content
            assert "tooltip" in content
            assert "breadcrumb" in content
            assert "navigateToRoot" in content
            assert "drawTreemap" in content
            assert "d3.hierarchy" in content
            assert "d3.treemap" in content

    def test_directory_creation_for_output_file(self):
        """Test that output directories are created if they don't exist."""
        timer = TimingStats()

        with timer.start("test"):
            time.sleep(0.001)

        timer.freeze()

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create nested path that doesn't exist
            output_file = Path(temp_dir) / "subdir" / "another" / "test.html"

            generate_d3_treemap_html(timer, str(output_file))

            assert output_file.exists()
            assert output_file.parent.exists()
