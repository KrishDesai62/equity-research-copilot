"""Local-first raw-data cache backed by an optional private S3 bucket."""
import os
from pathlib import Path

from dotenv import load_dotenv


class S3Cache:
    def __init__(self, bucket, prefix="equity-research", client=None):
        if not bucket:
            raise ValueError("An S3 bucket is required")
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        if client is None:
            import boto3
            client = boto3.client("s3")  # Standard credential chain, including AWS_PROFILE.
        self.client = client

    @classmethod
    def from_env(cls):
        if os.getenv("EQUITY_LOCAL_ONLY", "").lower() in {"1", "true", "yes"}:
            return None
        load_dotenv()
        if os.getenv("EQUITY_LOCAL_ONLY", "").lower() in {"1", "true", "yes"}:
            return None
        bucket = os.getenv("AWS_S3_BUCKET")
        return cls(bucket, os.getenv("AWS_S3_PREFIX", "equity-research")) if bucket else None

    def key(self, key):
        if key.startswith("/") or ".." in key.split("/"):
            raise ValueError("Cache keys must be relative and contain no parent traversal")
        return f"{self.prefix}/{key}" if self.prefix else key

    def restore(self, key, destination):
        from botocore.exceptions import ClientError
        try:
            result = self.client.get_object(Bucket=self.bucket, Key=self.key(key))
        except ClientError as exc:
            if exc.response["Error"]["Code"] in {"NoSuchKey", "404"}:
                return False
            raise  # Credentials/permissions errors are not cache misses.
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        body = result["Body"]
        try:
            data = body.read()
        finally:
            body.close()
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(data)
        temporary.replace(path)
        return True

    def upload(self, key, source):
        with Path(source).open("rb") as body:
            self.client.put_object(Bucket=self.bucket, Key=self.key(key), Body=body,
                                   ServerSideEncryption="AES256")
