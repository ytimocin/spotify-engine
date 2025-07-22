"""Benchmark performance improvements in data generation."""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime


def run_benchmark(users: int, songs: int, artists: int, days: int, runs: int = 3):
    """Run data generation benchmark multiple times and collect metrics."""
    times = []

    for i in range(runs):
        print(f"\nRun {i + 1}/{runs}...")
        start_time = time.time()

        # Run the data generation script
        cmd = [
            sys.executable,
            "scripts/generate_synthetic_data.py",
            "--users",
            str(users),
            "--songs",
            str(songs),
            "--artists",
            str(artists),
            "--days",
            str(days),
            "--seed",
            str(42 + i),  # Different seed for each run
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Error running data generation: {result.stderr}")
            continue

        end_time = time.time()
        elapsed = end_time - start_time
        times.append(elapsed)

        # Extract session count from output
        for line in result.stdout.split("\n"):
            if "Generated" in line and "listening sessions" in line:
                print(f"  {line.strip()}")

        print(f"  Time: {elapsed:.2f} seconds")

    if times:
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)

        return {
            "users": users,
            "songs": songs,
            "artists": artists,
            "days": days,
            "runs": runs,
            "times": times,
            "avg_time": avg_time,
            "min_time": min_time,
            "max_time": max_time,
        }

    return None


def main():
    """Main function to run performance benchmarks."""
    parser = argparse.ArgumentParser(description="Benchmark data generation performance")
    parser.add_argument("--small", action="store_true", help="Run small benchmark (100 users)")
    parser.add_argument("--medium", action="store_true", help="Run medium benchmark (1000 users)")
    parser.add_argument("--large", action="store_true", help="Run large benchmark (5000 users)")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs per benchmark")

    args = parser.parse_args()

    benchmarks = []

    if args.small or (not args.medium and not args.large):
        print("Running SMALL benchmark...")
        result = run_benchmark(100, 500, 50, 7, args.runs)
        if result:
            result["size"] = "small"
            benchmarks.append(result)

    if args.medium:
        print("\nRunning MEDIUM benchmark...")
        result = run_benchmark(1000, 5000, 500, 30, args.runs)
        if result:
            result["size"] = "medium"
            benchmarks.append(result)

    if args.large:
        print("\nRunning LARGE benchmark...")
        result = run_benchmark(5000, 10000, 1000, 30, args.runs)
        if result:
            result["size"] = "large"
            benchmarks.append(result)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"benchmark_results_{timestamp}.json"

    with open(filename, "w") as f:
        json.dump({"timestamp": timestamp, "benchmarks": benchmarks}, f, indent=2)

    # Print summary
    print("\n" + "=" * 50)
    print("BENCHMARK SUMMARY")
    print("=" * 50)

    for bench in benchmarks:
        print(f"\n{bench['size'].upper()} ({bench['users']} users, {bench['songs']} songs):")
        print(f"  Average time: {bench['avg_time']:.2f} seconds")
        print(f"  Min time: {bench['min_time']:.2f} seconds")
        print(f"  Max time: {bench['max_time']:.2f} seconds")

    print(f"\nResults saved to: {filename}")


if __name__ == "__main__":
    main()
