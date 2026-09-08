import subprocess
import sys
import os


def main():
    print("\n========================================")
    print("   ANGLE, DISTANCE AND VELOCITY MEASUREMENT")
    print("========================================")

    print("\nWhat do you want to measure?")
    print("1. Angle")
    print("2. Distance & Velocity")

    choice = input("\nEnter Angle/1 or Distance&Velocity/2: ").strip().lower()

    # Get the folder where main.py is located
    base_dir = os.path.dirname(os.path.abspath(__file__))

    if choice in ["angle", "1", "aoa"]:

        print("\nStarting ANGLE measurement...\n")

        aoa_file = os.path.join(base_dir, "aoa_kalman.py")

        subprocess.run(
            [sys.executable, aoa_file],
            check=True
        )

    elif choice in ["distance&velocity", "2", "range"]:

        print("\nStarting DISTANCE & VELOCITY measurement...\n")

        distance_file = os.path.join(base_dir, "dv_kalman.py")

        subprocess.run(
            [sys.executable, distance_file],
            check=True
        )

    else:
        print("\nInvalid input!")
        print("Please enter only: Angle or Distance&Velocity")


if __name__ == "__main__":
    main()