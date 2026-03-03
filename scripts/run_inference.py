import argparse

def main():
    parser = argparse.ArgumentParser(description="Run Multi-Subject 3D Pose Reconstruction Pipeline")
    parser.add_argument("--image", required=True, help="Path to input image")

    args = parser.parse_args()

    print(f"Running reconstruction on: {args.image}")
    print("Pipeline execution would happen here.")

if __name__ == "__main__":
    main()
