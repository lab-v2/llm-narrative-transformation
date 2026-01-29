#!/usr/bin/env python3
"""
Submit an Amazon Bedrock model customization (fine-tuning) job.

This script does NOT modify existing project code. It just submits a job.
It uses the AWS SDK's Bedrock control-plane API.
"""

import argparse
import json
import sys
from typing import Dict, Any, Optional


def _load_json_file(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Submit a Bedrock model customization job (fine-tuning)."
    )
    parser.add_argument("--job-name", required=True, help="Unique job name")
    parser.add_argument("--custom-model-name", required=True, help="Name for the resulting custom model")
    parser.add_argument("--base-model-id", required=True, help="Base model identifier (e.g., bedrock/us.meta.llama4-...)")
    parser.add_argument("--role-arn", required=True, help="IAM role ARN for Bedrock to access S3")
    parser.add_argument("--training-s3-uri", required=True, help="S3 URI to training data (e.g., s3://bucket/path/train.jsonl)")
    parser.add_argument("--output-s3-uri", required=True, help="S3 URI for output artifacts")
    parser.add_argument("--region", default=None, help="AWS region (overrides default if set)")
    parser.add_argument("--hyperparameters-json", default=None, help="Path to JSON file of hyperparameters")
    parser.add_argument("--tags-json", default=None, help="Path to JSON array of tags")
    args = parser.parse_args()

    hyperparameters = _load_json_file(args.hyperparameters_json) or {}
    tags = _load_json_file(args.tags_json) or []

    try:
        import boto3
    except ImportError:
        print("boto3 is required. Install with: pip install boto3", file=sys.stderr)
        return 2

    client = boto3.client("bedrock", region_name=args.region)

    request: Dict[str, Any] = {
        "jobName": args.job_name,
        "customModelName": args.custom_model_name,
        "baseModelIdentifier": args.base_model_id,
        "roleArn": args.role_arn,
        "trainingDataConfig": {
            "s3Uri": args.training_s3_uri,
        },
        "outputDataConfig": {
            "s3Uri": args.output_s3_uri,
        },
    }

    if hyperparameters:
        request["hyperParameters"] = hyperparameters
    if tags:
        request["tags"] = tags

    try:
        resp = client.create_model_customization_job(**request)
    except Exception as exc:
        print(f"Failed to create customization job: {exc}", file=sys.stderr)
        return 1

    job_arn = resp.get("jobArn") or resp.get("jobArn", "")
    print("Customization job submitted.")
    if job_arn:
        print(f"Job ARN: {job_arn}")
    else:
        print(resp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
