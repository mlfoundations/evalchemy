# SageMaker setup for LlamaFactory

The SageMaker setup here was done mostly with the TRI setting in mind, so it's possible that a few changes needed for other settings. The SageMaker launch script is at `dcft/train/sagemaker/launch_sagemaker_train.py`.

Sample usage:
```bash
python dcft/train/sagemaker/launch_sagemaker_train.py --user (user-id) --profile (aws-profile) --config dcft/train/configs/sample.yaml --instance-type p4de
```

Environment variables you may need to set:
- SAGEMAKER_ARN
- S3_REMOTE_SYNC -- S3 you want to sync results to. (Note: Keep in mind the region)
- HF_TOKEN
- DB_PASSWORD -- Ask in Slack if you need this.

The script is usable across regions as well, with the `--region` flag.

## Integration details
For a more thorough explanation of changes that needed to be done to integrate SageMaker, please see https://github.com/mlfoundations/dcft_private/pull/99.
