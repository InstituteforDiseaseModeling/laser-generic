#!/usr/bin/env python3
"""
Demonstration of execution order preservation in TimingStats.
Shows how timers are output in the order they were first encountered during execution,
not in alphabetical order.
"""

import time

from src.laser_generic.newutils import TimingStats as timer


def main():
    print("TimingStats Execution Order Demonstration")
    print("=" * 50)

    print("Creating timers in this execution order:")
    print("1. zebra_task")
    print("2. alpha_task")
    print("3. charlie_task")
    print("4. beta_task")
    print()

    # Create timers in non-alphabetical order
    with timer.start("zebra_task"):
        print("Executing zebra_task...")
        time.sleep(0.1)

        # Nested tasks also in non-alphabetical order
        with timer.start("zebra_subtask_2"):
            print("  Executing zebra_subtask_2...")
            time.sleep(0.02)

        with timer.start("zebra_subtask_1"):
            print("  Executing zebra_subtask_1...")
            time.sleep(0.01)

        with timer.start("zebra_subtask_3"):
            print("  Executing zebra_subtask_3...")
            time.sleep(0.03)

    with timer.start("alpha_task"):
        print("Executing alpha_task...")
        time.sleep(0.05)

    with timer.start("charlie_task"):
        print("Executing charlie_task...")
        time.sleep(0.08)

        with timer.start("charlie_preparation"):
            print("  Executing charlie_preparation...")
            time.sleep(0.02)

        with timer.start("charlie_execution"):
            print("  Executing charlie_execution...")
            time.sleep(0.03)

    with timer.start("beta_task"):
        print("Executing beta_task...")
        time.sleep(0.03)

    # Call some tasks again to show that execution order is based on first encounter
    print()
    print("Calling some tasks again (shouldn't change output order):")

    with timer.start("alpha_task"):
        print("  Re-executing alpha_task...")
        time.sleep(0.01)

    with timer.start("zebra_task"):
        print("  Re-executing zebra_task...")
        time.sleep(0.01)

    timer.freeze()

    print()
    print("Timing Results (showing execution order, NOT alphabetical):")
    print("-" * 60)
    print(timer.to_string())

    print()
    print("Notice how the output preserves execution order:")
    print("• Top-level: zebra_task, alpha_task, charlie_task, beta_task")
    print("• Zebra subtasks: zebra_subtask_2, zebra_subtask_1, zebra_subtask_3")
    print("• Charlie subtasks: charlie_preparation, charlie_execution")
    print()
    print("This would be alphabetical order:")
    print("• Top-level: alpha_task, beta_task, charlie_task, zebra_task")
    print("• Zebra subtasks: zebra_subtask_1, zebra_subtask_2, zebra_subtask_3")
    print("• Charlie subtasks: charlie_execution, charlie_preparation")


if __name__ == "__main__":
    main()
