import time
from unittest.mock import patch

import pytest

from laser_generic.newutils import _TimingStats


class TestTimingStats:
    def test_basic_initialization(self):
        timer = _TimingStats()
        assert not timer._frozen
        assert len(timer._timer_stack) == 0
        assert len(timer._timing_data) == 0

        return

    def test_freeze_functionality(self):
        timer = _TimingStats()
        timer.freeze()

        assert timer._frozen
        assert "__global__" in timer._timing_data
        assert timer._timing_data["__global__"]["call_count"] == 1

        return

    def test_cannot_start_after_freeze(self):
        timer = _TimingStats()
        timer.freeze()

        with pytest.raises(RuntimeError, match="Cannot start new timers after freeze"):
            timer.start("test_label")

        return

    def test_single_timer_context(self):
        timer = _TimingStats()

        with timer.start("test_task"):
            time.sleep(0.01)  # 10ms sleep

        timer.freeze()

        assert "test_task" in timer._timing_data
        data = timer._timing_data["test_task"]
        assert data["call_count"] == 1
        assert data["total_time"] > 0
        assert data["parent"] is None
        assert len(data["children"]) == 0

        return

    def test_nested_timers(self):
        timer = _TimingStats()

        with timer.start("outer"):
            time.sleep(0.005)  # 5ms
            with timer.start("inner1"):
                time.sleep(0.005)  # 5ms
            with timer.start("inner2"):
                time.sleep(0.005)  # 5ms

        timer.freeze()

        # Check that both timers exist (now using path-based keys)
        assert "outer" in timer._timing_data
        assert "outer/inner1" in timer._timing_data
        assert "outer/inner2" in timer._timing_data

        # Check hierarchy
        outer_data = timer._timing_data["outer"]
        inner1_data = timer._timing_data["outer/inner1"]
        inner2_data = timer._timing_data["outer/inner2"]

        assert inner1_data["parent"] == "outer"
        assert inner2_data["parent"] == "outer"
        assert "outer/inner1" in outer_data["children"]
        assert "outer/inner2" in outer_data["children"]
        assert outer_data["parent"] is None

        # Check that outer time >= inner times combined
        total_inner_time = inner1_data["total_time"] + inner2_data["total_time"]
        assert outer_data["total_time"] >= total_inner_time

        return

    def test_multiple_calls_same_label(self):
        timer = _TimingStats()

        for _ in range(3):
            with timer.start("repeated_task"):
                time.sleep(0.001)  # 1ms

        timer.freeze()

        assert "repeated_task" in timer._timing_data
        data = timer._timing_data["repeated_task"]
        assert data["call_count"] == 3
        assert data["total_time"] > 0

        return

    def test_string_output_different_scales(self):
        timer = _TimingStats()

        with timer.start("test_task"):
            time.sleep(0.01)  # 10ms

        timer.freeze()

        # Test different scales
        ms_output = timer.to_string(scale="ms")
        assert "test_task:" in ms_output
        assert "ms" in ms_output

        ns_output = timer.to_string(scale="ns")
        assert "test_task:" in ns_output
        assert "ns" in ns_output

        s_output = timer.to_string(scale="s")
        assert "test_task:" in s_output
        assert "s" in s_output

        return

    def test_string_output_before_freeze_raises_error(self):
        timer = _TimingStats()

        with timer.start("test_task"):
            time.sleep(0.01)

        with pytest.raises(RuntimeError, match="Must call freeze"):
            timer.to_string()

        return

    def test_hierarchical_string_output(self):
        timer = _TimingStats()

        with timer.start("parent"):
            time.sleep(0.001)
            with timer.start("child1"):
                time.sleep(0.001)
            with timer.start("child2"):
                time.sleep(0.001)

        timer.freeze()
        output = timer.to_string(scale="ms")

        lines = output.split("\n")
        assert len(lines) >= 3

        # Check indentation for hierarchy
        parent_line = next(line for line in lines if "parent:" in line)
        child1_line = next(line for line in lines if "child1:" in line)
        child2_line = next(line for line in lines if "child2:" in line)

        assert not parent_line.startswith("  ")  # No indent for parent
        assert child1_line.startswith("  ")  # Indent for child
        assert child2_line.startswith("  ")  # Indent for child

        return

    def test_self_time_calculation(self):
        timer = _TimingStats()

        with timer.start("parent"):
            time.sleep(0.005)  # 5ms self time
            with timer.start("child"):
                time.sleep(0.005)  # 5ms child time

        timer.freeze()

        parent_data = timer._timing_data["parent"]
        child_data = timer._timing_data["parent/child"]

        # Parent's self time should be less than total time due to child
        assert parent_data["self_time"] < parent_data["total_time"]
        assert parent_data["self_time"] >= 0

        # Child's self time should equal total time (no nested children)
        assert abs(child_data["self_time"] - child_data["total_time"]) < 1000000  # 1ms tolerance in ns

        return

    def test_invalid_scale_raises_error(self):
        timer = _TimingStats()

        with timer.start("test"):
            pass

        timer.freeze()

        with pytest.raises(ValueError, match="Invalid scale"):
            timer.to_string(scale="invalid")

        return

    @patch("matplotlib.pyplot.show")
    def test_plot_treemap_functionality(self, mock_show):
        timer = _TimingStats()

        with timer.start("parent"):
            time.sleep(0.001)
            with timer.start("child1"):
                time.sleep(0.001)
            with timer.start("child2"):
                time.sleep(0.001)

        timer.freeze()

        # Should not raise an error
        timer.plot_treemap(title="Test Treemap")
        mock_show.assert_called_once()

        return

    def test_plot_treemap_before_freeze_raises_error(self):
        timer = _TimingStats()

        with timer.start("test"):
            pass

        with pytest.raises(RuntimeError, match="Must call freeze"):
            timer.plot_treemap()

        return

    def test_deeply_nested_timers(self):
        timer = _TimingStats()

        with timer.start("level1"):
            time.sleep(0.001)
            with timer.start("level2"):
                time.sleep(0.001)
                with timer.start("level3"):
                    time.sleep(0.001)
                    with timer.start("level4"):
                        time.sleep(0.001)

        timer.freeze()

        # Verify all levels exist and have correct parents
        assert timer._timing_data["level1"]["parent"] is None
        assert timer._timing_data["level2"]["parent"] == "level1"
        assert timer._timing_data["level3"]["parent"] == "level2"
        assert timer._timing_data["level4"]["parent"] == "level3"

        # Check children relationships
        assert "level2" in timer._timing_data["level1"]["children"]
        assert "level3" in timer._timing_data["level2"]["children"]
        assert "level4" in timer._timing_data["level3"]["children"]
        assert len(timer._timing_data["level4"]["children"]) == 0

        return

    def test_parallel_nested_structure(self):
        timer = _TimingStats()

        with timer.start("root"):
            with timer.start("branch_a"):
                time.sleep(0.001)
                with timer.start("leaf_a1"):
                    time.sleep(0.001)
                with timer.start("leaf_a2"):
                    time.sleep(0.001)

            with timer.start("branch_b"):
                time.sleep(0.001)
                with timer.start("leaf_b1"):
                    time.sleep(0.001)

        timer.freeze()

        # Verify structure
        root_data = timer._timing_data["root"]
        assert "branch_a" in root_data["children"]
        assert "branch_b" in root_data["children"]

        branch_a_data = timer._timing_data["branch_a"]
        assert "leaf_a1" in branch_a_data["children"]
        assert "leaf_a2" in branch_a_data["children"]

        branch_b_data = timer._timing_data["branch_b"]
        assert "leaf_b1" in branch_b_data["children"]

        return

    def test_empty_timer_stats(self):
        timer = _TimingStats()
        timer.freeze()

        # Should only have global timer
        assert "__global__" in timer._timing_data
        assert len(timer._timing_data) == 1

        output = timer.to_string()
        assert "__global__" in output

        return

    def test_context_manager_exception_handling(self):
        timer = _TimingStats()

        # Test that timer still works properly even if exception occurs
        try:
            with timer.start("error_task"):
                time.sleep(0.001)
                raise ValueError("Test error")
        except ValueError:
            pass  # Expected

        timer.freeze()

        # Timer should still have recorded the task
        assert "error_task" in timer._timing_data
        assert timer._timing_data["error_task"]["call_count"] == 1
        assert timer._timing_data["error_task"]["total_time"] > 0

        return

    def test_execution_order_preservation(self):
        """Test that timers are output in execution order, not alphabetical order."""
        timer = _TimingStats()

        # Create timers in a specific order that differs from alphabetical
        with timer.start("zebra"):
            time.sleep(0.001)
        with timer.start("alpha"):
            time.sleep(0.001)
        with timer.start("beta"):
            time.sleep(0.001)
        with timer.start("charlie"):
            time.sleep(0.001)

        timer.freeze()
        output = timer.to_string()
        lines = output.split("\n")

        # Find the order of appearance in output
        zebra_line = next(i for i, line in enumerate(lines) if "zebra:" in line)
        alpha_line = next(i for i, line in enumerate(lines) if "alpha:" in line)
        beta_line = next(i for i, line in enumerate(lines) if "beta:" in line)
        charlie_line = next(i for i, line in enumerate(lines) if "charlie:" in line)

        # Should be in execution order: zebra, alpha, beta, charlie
        # Not alphabetical order: alpha, beta, charlie, zebra
        assert zebra_line < alpha_line < beta_line < charlie_line

        return

    def test_nested_execution_order_preservation(self):
        """Test that nested timers are output in execution order within their level."""
        timer = _TimingStats()

        with timer.start("parent"):
            with timer.start("child_zebra"):
                time.sleep(0.001)
            with timer.start("child_alpha"):
                time.sleep(0.001)
            with timer.start("child_beta"):
                time.sleep(0.001)

        timer.freeze()
        output = timer.to_string()
        lines = output.split("\n")

        # Find child lines (they should have indentation)
        child_zebra_line = next(i for i, line in enumerate(lines) if "child_zebra:" in line and line.startswith("  "))
        child_alpha_line = next(i for i, line in enumerate(lines) if "child_alpha:" in line and line.startswith("  "))
        child_beta_line = next(i for i, line in enumerate(lines) if "child_beta:" in line and line.startswith("  "))

        # Should be in execution order: child_zebra, child_alpha, child_beta
        assert child_zebra_line < child_alpha_line < child_beta_line

        return

    def test_execution_order_tracks_first_encounter(self):
        """Test that execution order is based on first encounter, not subsequent calls."""
        timer = _TimingStats()

        # First encounters in this order
        with timer.start("task_c"):
            time.sleep(0.001)

        with timer.start("task_a"):
            time.sleep(0.001)

        with timer.start("task_b"):
            time.sleep(0.001)

        # Call them again in different order - shouldn't change execution order
        with timer.start("task_a"):
            time.sleep(0.001)

        with timer.start("task_c"):
            time.sleep(0.001)

        with timer.start("task_b"):
            time.sleep(0.001)

        timer.freeze()

        # Check internal execution order data
        assert timer._timing_data["task_c"]["execution_order"] == 1
        assert timer._timing_data["task_a"]["execution_order"] == 2
        assert timer._timing_data["task_b"]["execution_order"] == 3

        # Check output order
        output = timer.to_string()
        lines = output.split("\n")

        task_c_line = next(i for i, line in enumerate(lines) if "task_c:" in line)
        task_a_line = next(i for i, line in enumerate(lines) if "task_a:" in line)
        task_b_line = next(i for i, line in enumerate(lines) if "task_b:" in line)

        assert task_c_line < task_a_line < task_b_line

        return

    def test_mixed_level_execution_order(self):
        """Test execution order with mixed nesting levels."""
        timer = _TimingStats()

        with timer.start("root1"):
            time.sleep(0.001)
            with timer.start("child1_1"):
                time.sleep(0.001)

        with timer.start("root2"):
            time.sleep(0.001)

        with timer.start("root3"):
            time.sleep(0.001)
            with timer.start("child3_1"):
                time.sleep(0.001)
            with timer.start("child3_2"):
                time.sleep(0.001)

        timer.freeze()

        # Verify execution order tracking
        assert timer._timing_data["root1"]["execution_order"] == 1
        assert timer._timing_data["child1_1"]["execution_order"] == 2
        assert timer._timing_data["root2"]["execution_order"] == 3
        assert timer._timing_data["root3"]["execution_order"] == 4
        assert timer._timing_data["root3/child3_1"]["execution_order"] == 5
        assert timer._timing_data["root3/child3_2"]["execution_order"] == 6

        return

    def test_same_name_different_parents_no_coalescing(self):
        """Test the bug fix: contexts with same name under different parents should not be coalesced."""
        timer = _TimingStats()

        # Two different code paths both call "process"
        with timer.start("parent_a"):
            time.sleep(0.001)
            with timer.start("process"):
                time.sleep(0.002)

        with timer.start("parent_b"):
            time.sleep(0.001)
            with timer.start("process"):
                time.sleep(0.003)

        timer.freeze()

        # Should have separate entries for each process context
        assert "parent_a/process" in timer._timing_data
        assert "parent_b/process" in timer._timing_data

        # Each should have call_count of 1, not coalesced
        parent_a_process = timer._timing_data["parent_a/process"]
        parent_b_process = timer._timing_data["parent_b/process"]

        assert parent_a_process["call_count"] == 1
        assert parent_b_process["call_count"] == 1

        # Each parent should have their own process child
        parent_a_data = timer._timing_data["parent_a"]
        parent_b_data = timer._timing_data["parent_b"]

        assert "parent_a/process" in parent_a_data["children"]
        assert "parent_b/process" in parent_b_data["children"]

        # Parent self-times should be positive (not negative due to double-counting)
        assert parent_a_data["self_time"] >= 0
        assert parent_b_data["self_time"] >= 0

        # Display should show correct labels (not full paths)
        output = timer.to_string()
        lines = output.split("\n")

        # Should have two separate "process" entries under different parents
        process_lines = [line for line in lines if "process:" in line and line.startswith("  ")]
        assert len(process_lines) == 2

        return
