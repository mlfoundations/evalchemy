import os
import shlex
import subprocess

from setuptools import find_packages, setup


def install_requirements(requirements_path):
    with open(requirements_path, "r") as f:
        for line in f:
            # Remove version specifiers and comments
            package = line.split("==")[0].split(">")[0].split("<")[0].split("#")[0].strip()
            if package:
                print(f"Installing {package}")
                # Remove quotes from package name since this breaks pip
                package = package.replace('"', "")
                subprocess.check_call(["pip", "install", package])


def run_setup_files_and_install_requirements():
    base_path = "dcft/external_repositories"
    for root, dirs, files in os.walk(base_path):
        if "setup.py" in files:
            try:
                original_dir = os.getcwd()
                print(f"Running setup.py in {root}")
                os.chdir(root)
                subprocess.check_call(["python", "setup.py", "install"])
            finally:
                os.chdir(original_dir)

        if "requirements.txt" in files:
            req_path = os.path.join(root, "requirements.txt")
            print(f"Installing requirements from {req_path}")
            install_requirements(req_path)


setup(
    name="dcft-external-repositories",
    version="0.1",
    packages=find_packages(),
    install_requires=[],
    entry_points={
        "console_scripts": [
            "install_external_repos=setup:run_setup_files_and_install_requirements",
        ],
    },
)
if __name__ == "__main__":
    run_setup_files_and_install_requirements()
