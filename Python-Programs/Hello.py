import os
import sys

def main():
    print("=== Greeting ===")
    print("Hello, Partap!")

    print("\n=== System Info ===")
    print("Python version:", sys.version)
    print("Executable path:", sys.executable)

    print("\n=== Current Directory ===")
    print("Working directory:", os.getcwd())

if __name__ == "__main__":
    main()
