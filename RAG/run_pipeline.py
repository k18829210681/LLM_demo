#!/usr/bin/env python3
"""
RAG Dataset Generation Pipeline Runner

Runs all pipeline steps sequentially based on config.yaml.

Usage:
  python run_pipeline.py           # Run full pipeline
  python run_pipeline.py --test    # Run in test mode (faster, smaller datasets)
"""

from __future__ import annotations

import os
import sys
import time

# Import all step modules (using numbered names)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import importlib
step1_chunking = importlib.import_module("1_chunking")
step2_deduplication = importlib.import_module("2_deduplication")
step3_qa_generation = importlib.import_module("3_qa_generation")
step4_qa_judging = importlib.import_module("4_qa_judging")
step5_embedding = importlib.import_module("5_embedding")
step6_reranking = importlib.import_module("6_reranking")
step7_benchmark = importlib.import_module("7_benchmark")


def run_step(step_num: int, step_name: str, step_module) -> bool:
    """Run a single pipeline step and handle errors."""
    print("\n" + "=" * 70)
    print(f"STEP {step_num}: {step_name}")
    print("=" * 70)
    print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    start_time = time.time()
    try:
        step_module.main()
        elapsed = time.time() - start_time
        print(f"\n✓ Step {step_num} completed successfully in {elapsed:.1f}s")
        return True
    except KeyboardInterrupt:
        print(f"\n✗ Step {step_num} interrupted by user")
        raise
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n✗ Step {step_num} failed after {elapsed:.1f}s")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run the complete pipeline."""
    # Check for test mode
    test_mode = "--test" in sys.argv

    if test_mode:
        print("\n" + "=" * 70)
        print("⚠ RUNNING IN TEST MODE")
        print("=" * 70)
        print("This will use smaller datasets for faster testing.")
        print("To run full pipeline, omit --test flag.")
        print("=" * 70)
        os.environ["TESTING_MODE"] = "1"

    print("\n" + "=" * 70)
    print("RAG DATASET GENERATION PIPELINE")
    print("=" * 70)
    print(f"Pipeline started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Test mode: {'ENABLED' if test_mode else 'DISABLED'}")
    print("=" * 70)

    pipeline_start = time.time()

    steps = [
        (1, "Chunking", step1_chunking),
        (2, "Deduplication", step2_deduplication),
        (3, "QA Generation", step3_qa_generation),
        (4, "QA Quality Judging", step4_qa_judging),
        (5, "Embedding", step5_embedding),
        (6, "Reranking", step6_reranking),
        (7, "Benchmark", step7_benchmark),
    ]

    completed_steps = []
    failed_step = None

    for step_num, step_name, step_module in steps:
        success = run_step(step_num, step_name, step_module)
        if success:
            completed_steps.append((step_num, step_name))
        else:
            failed_step = (step_num, step_name)
            break

    # Print final summary
    pipeline_elapsed = time.time() - pipeline_start
    print("\n" + "=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)

    if failed_step:
        print(f"\n✗ Pipeline FAILED at step {failed_step[0]}: {failed_step[1]}")
        print(f"\nCompleted steps ({len(completed_steps)}/{len(steps)}):")
        for num, name in completed_steps:
            print(f"  ✓ Step {num}: {name}")
        print(f"\nTotal time: {pipeline_elapsed:.1f}s ({pipeline_elapsed/60:.1f} minutes)")
        sys.exit(1)
    else:
        print(f"\n✓ Pipeline completed successfully!")
        print(f"\nAll steps completed ({len(steps)}/{len(steps)}):")
        for num, name in completed_steps:
            print(f"  ✓ Step {num}: {name}")
        print(f"\nTotal time: {pipeline_elapsed:.1f}s ({pipeline_elapsed/60:.1f} minutes)")
        print("\n" + "=" * 70)
        print("Pipeline finished at:", time.strftime("%Y-%m-%d %H:%M:%S"))
        print("=" * 70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user. Exiting...")
        sys.exit(1)
