#!/usr/bin/env python3
"""
Demonstration of D3.js HTML treemap generation for TimingStats.
This creates an interactive HTML file that can be opened in a web browser.
"""

import time

from src.laser_generic.newutils import TimingStats
from src.laser_generic.tstreemap import generate_d3_treemap_html


def simulate_complex_workflow():
    """Simulate a complex workflow with nested timing operations."""
    timer = TimingStats()

    print("Starting complex workflow simulation...")

    with timer.start("data_pipeline"):
        print("  Running data pipeline...")
        time.sleep(0.1)

        with timer.start("data_ingestion"):
            print("    Data ingestion...")
            time.sleep(0.05)

            with timer.start("file_discovery"):
                print("      File discovery...")
                time.sleep(0.01)

            with timer.start("file_validation"):
                print("      File validation...")
                time.sleep(0.02)

            with timer.start("data_loading"):
                print("      Data loading...")
                time.sleep(0.02)

        with timer.start("data_processing"):
            print("    Data processing...")
            time.sleep(0.08)

            with timer.start("data_cleansing"):
                print("      Data cleansing...")
                time.sleep(0.03)

            with timer.start("feature_engineering"):
                print("      Feature engineering...")
                time.sleep(0.02)

            with timer.start("data_validation"):
                print("      Data validation...")
                time.sleep(0.03)

        with timer.start("model_training"):
            print("    Model training...")
            time.sleep(0.12)

            with timer.start("hyperparameter_tuning"):
                print("      Hyperparameter tuning...")
                time.sleep(0.04)

            with timer.start("model_fitting"):
                print("      Model fitting...")
                time.sleep(0.05)

            with timer.start("model_evaluation"):
                print("      Model evaluation...")
                time.sleep(0.03)

    with timer.start("results_processing"):
        print("  Results processing...")
        time.sleep(0.06)

        with timer.start("metrics_calculation"):
            print("    Metrics calculation...")
            time.sleep(0.02)

        with timer.start("visualization_generation"):
            print("    Visualization generation...")
            time.sleep(0.02)

        with timer.start("report_generation"):
            print("    Report generation...")
            time.sleep(0.02)

    # Multiple calls to some operations
    for i in range(3):
        with timer.start("cleanup_operations"):
            print(f"    Cleanup operation {i + 1}...")
            time.sleep(0.01)

    timer.freeze()
    print("Workflow simulation completed!")
    return timer


def main():
    print("D3.js Interactive Treemap Demo")
    print("=" * 40)

    # Create timing data
    timer = simulate_complex_workflow()

    # Show text summary first
    print("\nTiming Summary:")
    print("-" * 30)
    print(timer.to_string(scale="ms"))

    # Generate D3 treemap HTML files with different configurations
    print("\nGenerating interactive D3 treemap visualizations...")

    # Standard treemap
    generate_d3_treemap_html(timer, "timing_treemap_standard.html", title="Workflow Execution Treemap", scale="ms", width=1200, height=800)
    print("✓ Created: timing_treemap_standard.html")

    # Compact treemap
    generate_d3_treemap_html(timer, "timing_treemap_compact.html", title="Compact Timing Analysis", scale="ms", width=800, height=600)
    print("✓ Created: timing_treemap_compact.html")

    # High precision treemap
    generate_d3_treemap_html(
        timer, "timing_treemap_microseconds.html", title="High Precision Timing Analysis", scale="microseconds", width=1400, height=900
    )
    print("✓ Created: timing_treemap_microseconds.html")

    print("\n" + "=" * 60)
    print("Interactive D3 Treemap Files Generated!")
    print("=" * 60)
    print()
    print("Features of the generated HTML files:")
    print("• Interactive navigation - click on sections to drill down")
    print("• Hover tooltips showing detailed timing information")
    print("• Breadcrumb navigation to move back up the hierarchy")
    print("• Color-coded depth levels for easy visual distinction")
    print("• Responsive design that works on different screen sizes")
    print("• Self-contained HTML files (no external dependencies except D3.js CDN)")
    print()
    print("To view the treemaps:")
    print("1. Open any of the generated .html files in your web browser")
    print("2. Hover over sections to see detailed timing information")
    print("3. Click on parent sections to navigate into their children")
    print("4. Use the breadcrumb navigation to go back to parent levels")
    print()
    print("The treemap visualization shows:")
    print("• Rectangle size = proportional to execution time")
    print("• Color = indicates nesting depth")
    print("• Text labels = show operation names and timing values")
    print("• Interactive stats = total time, calls, and component counts")

    # Show some statistics
    total_operations = len(timer._timing_data)
    total_time_ms = sum(data["total_time"] for data in timer._timing_data.values() if data["parent"] is None) / 1_000_000

    print()
    print("Workflow Statistics:")
    print(f"• Total operations tracked: {total_operations}")
    print(f"• Total execution time: {total_time_ms:.2f}ms")
    print("• Deepest nesting level: 3 (data_pipeline → data_ingestion → file_discovery)")
    print("• Most called operation: cleanup_operations (3 calls)")


if __name__ == "__main__":
    main()
