import argparse
import time
import os
import subprocess
import yaml
from datetime import datetime
from pathlib import Path
import boto3
import sagemaker
from sagemaker.pytorch import PyTorch

NAME = "dcft-train"
INSTANCE_MAPPER = {
    "p4": "ml.p4d.24xlarge",
    "p4de": "ml.p4de.24xlarge",
    "p5": "ml.p5.48xlarge",
}
QUEUE_MAPPER = {
    "us-east-1": {
        "ml.p5.48xlarge": "fss-ml-p5-48xlarge-us-east-1",
        "ml.p4de.24xlarge": "fss-ml-p4de-24xlarge-us-east-1",
    },
    "us-west-2": {
        "ml.p4de.24xlarge": "fss-ml-p4de-24xlarge-us-west-2",
        "ml.p4d.24xlarge": "fss-ml-p4d-24xlarge-us-west-2",
    },
}


def run_command(command):
    print(f"=> {command}")
    subprocess.run(command, shell=True, check=True)


def get_image(user, profile="default", region="us-east-1"):
    os.environ["AWS_PROFILE"] = f"{profile}"
    account = subprocess.getoutput(
        f"aws --region {region} --profile {profile} sts get-caller-identity --query Account --output text"
    )
    docker_dir = Path(__file__).parent
    algorithm_name = f"{user}-{NAME}"
    dockerfile_base = docker_dir / "Dockerfile"
    fullname = f"{account}.dkr.ecr.{region}.amazonaws.com/{algorithm_name}:latest"

    login_cmd = f"aws ecr get-login-password --region {region} --profile {profile} | docker login --username AWS --password-stdin"

    print("Building container")
    commands = [
        # Log in to Sagemaker account to get image.
        f"{login_cmd} 763104351884.dkr.ecr.{region}.amazonaws.com",
        f"docker build --progress=plain -f {dockerfile_base} --build-arg AWS_REGION={region} -t {algorithm_name} .",
        f"docker tag {algorithm_name} {fullname}",
        f"{login_cmd} {fullname}",
        (
            f"aws --region {region} ecr describe-repositories --repository-names {algorithm_name} || "
            f"aws --region {region} ecr create-repository --repository-name {algorithm_name}"
        ),
    ]

    # Create command, making sure to exit if any part breaks.
    command = "\n".join([f"{x} || exit 1" for x in commands])
    run_command(command)
    run_command(f"docker push {fullname}")
    print("Sleeping for 5 seconds to ensure push succeeded")
    time.sleep(5)
    return f"{account}.dkr.ecr.{region}.amazonaws.com/{algorithm_name}:latest"


def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--user", required=True, help="User name")
    parser.add_argument("--config", required=True, help="Model yaml")

    # AWS profile args
    parser.add_argument("--region", default="us-east-1", help="AWS region")
    parser.add_argument("--profile", default="default", help="AWS profile to use")
    parser.add_argument("--arn", default=None, help="If None, reads from SAGEMAKER_ARN env var")
    parser.add_argument(
        "--s3-remote-sync", default=None, help="S3 path to sync to. If none, reads from S3_REMOTE_SYNC env var"
    )

    # Instance args
    parser.add_argument("--instance-count", default=1, type=int, help="Number of instances")
    parser.add_argument("--instance-type", default="p4de", choices=list(INSTANCE_MAPPER.keys()))
    parser.add_argument("--spot-instance", action="store_true")

    # SageMaker queue args
    parser.add_argument("--use-queue", action="store_true")
    parser.add_argument("--priority", type=int, default=10, help="SageMaker FSS queue priority")

    args = parser.parse_args()
    main_after_setup_move(args)


def main_after_setup_move(args):
    assert args.instance_type in INSTANCE_MAPPER
    if args.arn is None:
        assert "SAGEMAKER_ARN" in os.environ, "Please specify --arn or set the SAGEMAKER_ARN environment variable"
        args.arn = os.environ["SAGEMAKER_ARN"]

    if args.s3_remote_sync is None:
        assert (
            "S3_REMOTE_SYNC" in os.environ
        ), "Please specify --s3-remote-sync or set the S3_REMOTE_SYNC environment variable"
        args.s3_remote_sync = os.environ["S3_REMOTE_SYNC"]
        args.s3_remote_sync = args.s3_remote_sync.replace("us-east-1", args.region)

    image = get_image(
        args.user,
        region=args.region,
        profile=args.profile,
    )

    ##########
    # Create session and make sure of account and region
    ##########
    sagemaker_session = sagemaker.Session(boto_session=boto3.session.Session(region_name=args.region))

    if args.local:
        from sagemaker.local import LocalSession

        sagemaker_session = LocalSession()

    role = args.arn
    # provide a pre-existing role ARN as an alternative to creating a new role
    role_name = role.split(["/"][-1])
    print(f"SageMaker Execution Role:{role}")
    print(f"The name of the Execution role: {role_name[-1]}")

    client = boto3.client("sts")
    account = client.get_caller_identity()["Account"]
    print(f"AWS account:{account}")

    session = boto3.session.Session()
    region = session.region_name
    print(f"AWS region:{region}")

    ##########
    # Configure the training
    ##########
    base_job_name = f"{args.user.replace('.', '-')}-{NAME}"

    checkpoint_local_path = "/opt/ml/checkpoints"

    def get_job_name(base):
        now = datetime.now()
        # Format example: 2023-03-03-10-14-02-324
        now_ms_str = f"{now.microsecond // 1000:03d}"
        date_str = f"{now.strftime('%Y-%m-%d-%H-%M-%S')}-{now_ms_str}"
        job_name = "_".join([base, date_str])
        return job_name

    job_name = get_job_name(base_job_name)

    output_root = f"{args.s3_remote_sync}/sagemaker/{args.user}/{NAME}/"
    output_s3 = os.path.join(output_root, job_name)

    with open(args.config, "r") as f:
        hyperparameters = yaml.safe_load(f)
    hyperparameters["deepspeed"] = hyperparameters["deepspeed"].replace("dcft/train", "/opt/ml/code")
    hyperparameters["enable_liger_kernel"] = False
    hyperparameters["output_dir"] = "/opt/ml/checkpoints"

    environment = {
        "SM_USE_RESERVED_CAPACITY": "1",
        "HF_TOKEN": os.environ["HF_TOKEN"],
        "DB_PASSWORD": os.environ["DB_PASSWORD"],
    }
    estimator = PyTorch(
        entry_point="dcft/train/llamafactory/src/train.py",
        sagemaker_session=sagemaker_session,
        base_job_name=base_job_name,
        hyperparameters=hyperparameters,
        role=role,
        image_uri=image,
        instance_count=args.instance_count,
        instance_type="local_gpu" if args.local else INSTANCE_MAPPER[args.instance_type],
        train_use_spot_instances=args.spot_instance,
        output_path=output_s3,
        job_name=job_name,
        checkpoint_s3_uri=None if args.local else f"{output_s3}/checkpoint",
        checkpoint_local_path=None if args.local else checkpoint_local_path,
        code_location=output_s3,
        # Training using SMDataParallel Distributed Training Framework
        distribution={"torch_distributed": {"enabled": True}},
        # Max run 5 days
        max_run=5 * 24 * 60 * 60,
        max_wait=5 * 24 * 60 * 60 if args.spot_instance else None,
        input_mode="FastFile",
        environment=environment,
        keep_alive_period_in_seconds=30 * 60 if not args.spot_instance else None,  # 30 minutes
        tags=[
            {"Key": "tri.project", "Value": "MM:PJ-0077"},
            {"Key": "tri.owner.email", "Value": f"{args.user}@tri.global"},
        ],
    )

    if args.use_queue:
        from sagemaker.batch_queueing.queue import Queue

        queue = Queue(queue_name=QUEUE_MAPPER[args.region][args.instance_type])
        queued_jobs = queue.map(
            estimator, inputs=[None], job_names=[job_name], priority=args.priority, share_identifier="default"
        )
        print(f"Queued {job_name}")
    else:
        estimator.fit()


if __name__ == "__main__":
    main()
