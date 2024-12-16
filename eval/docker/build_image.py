import os
import subprocess
import time
from pathlib import Path

NAME = "evalchemy"


def run_command(command):
    """
    Executes a shell command and checks for errors.

    Args:
        command (str): The shell command to execute.
    """
    print(f"=> {command}")
    subprocess.run(command, shell=True, check=True)


def get_image(user, profile="default", region="us-west-2"):
    """
    Builds and pushes a Docker image to AWS ECR for the specified user and instance type.

    Args:
        user (str): The user for whom the image is being built (e.g., "firstname.lastname").
        profile (str, optional): AWS profile to use. Defaults to "default".
        region (str, optional): AWS region to use. Defaults to "us-west-2".

    Returns:
        str: The full name of the Docker image in ECR.
    """
    os.environ["AWS_PROFILE"] = profile
    os.environ["AWS_REGION"] = region
    account = subprocess.getoutput(
        f"aws --region {region} --profile {profile} sts get-caller-identity --query Account --output text"
    )
    docker_dir = Path(__file__).parent

    algorithm_name = f"{user}-{NAME}"
    dockerfile_base = docker_dir / "Dockerfile"
    fullname = f"{account}.dkr.ecr.{region}.amazonaws.com/{algorithm_name}:latest"

    # Calculate hash of local files
    local_hash = subprocess.getoutput(
        f"find . -type f -not -path '*/\.*' -exec sha256sum {{}} \; | sort | sha256sum | cut -d' ' -f1"
    )

    # Try to get the hash from the existing image
    existing_hash = subprocess.getoutput(
        f"docker pull {fullname} 2>/dev/null && "
        f"docker inspect {fullname} --format='{{{{.Config.Labels.code_hash}}}}' 2>/dev/null"
    )

    if existing_hash and existing_hash == local_hash:
        print("Found existing image with identical code, skipping build")
        return fullname

    login_cmd = (
        f"aws ecr get-login-password --region {region} --profile {profile} |     docker login --username AWS"
        " --password-stdin"
    )

    print("Building container")
    commands = [
        # Log in to Sagemaker account to get image.
        f"{login_cmd} 763104351884.dkr.ecr.{region}.amazonaws.com",
        f"docker build --progress=plain -f {dockerfile_base} --build-arg AWS_REGION={region} "
        f"--label code_hash={local_hash} -t {algorithm_name} .",
        f"docker tag {algorithm_name} {fullname}",
        f"{login_cmd} {fullname}",
        f"aws --region {region} ecr describe-repositories --repository-names {algorithm_name}  --no-cli-pager  || "
        f"aws --region {region} ecr create-repository --repository-name {algorithm_name}  --no-cli-pager ",
    ]

    # Create command, making sure to exit if any part breaks.
    command = "\n".join([f"{x} || exit 1" for x in commands])
    run_command(command)
    run_command(f"docker push {fullname}")
    print("Sleeping for 5 seconds to ensure push succeeded")
    time.sleep(5)
    return f"{account}.dkr.ecr.{region}.amazonaws.com/{algorithm_name}:latest"
