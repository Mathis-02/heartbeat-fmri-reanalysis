from pathlib import Path
import argparse
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "subjects",
        nargs="+",
        help="Subjects to process, e.g. sub-09261 sub-09301",
    )

    args = parser.parse_args()

    successes = []
    failures = []

    total_start = time.perf_counter()

    for index, subject in enumerate(args.subjects, start=1):

        print("\n" + "=" * 80)
        print(
            f"[{index}/{len(args.subjects)}] "
            f"PROCESSING {subject}"
        )
        print("=" * 80)

        start = time.perf_counter()

        result = subprocess.run(
            [
                sys.executable,
                "scripts/run_pipeline.py",
                subject,
            ],
            check=False,
        )

        duration = time.perf_counter() - start

        if result.returncode == 0:
            successes.append(subject)

            print(
                f"\nSUCCESS: {subject} "
                f"({duration / 60:.1f} min)"
            )

        else:
            failures.append({
                "subject": subject,
                "return_code": result.returncode,
            })

            print(
                f"\nFAILED: {subject} "
                f"(return code {result.returncode})"
            )

    total_duration = time.perf_counter() - total_start

    print("\n" + "=" * 80)
    print("BATCH SUMMARY")
    print("=" * 80)

    print(
        f"Total duration: "
        f"{total_duration / 60:.1f} min"
    )

    print(
        f"Successful: "
        f"{len(successes)}/{len(args.subjects)}"
    )

    for subject in successes:
        print(f"  OK     {subject}")

    print(
        f"Failed: "
        f"{len(failures)}/{len(args.subjects)}"
    )

    for failure in failures:
        print(
            f"  FAILED {failure['subject']} "
            f"(code {failure['return_code']})"
        )

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()