#!/usr/bin/env node
import * as cdk from "aws-cdk-lib";
import { NnmctsPipelineStack } from "../lib/nnmcts-pipeline-stack";

const app = new cdk.App();

const account = process.env.CDK_DEFAULT_ACCOUNT;
const region = process.env.CDK_DEFAULT_REGION ?? "us-west-1";

if (!account) {
  throw new Error(
    "CDK_DEFAULT_ACCOUNT is required. Export it from your AWS CLI identity, e.g. " +
      "CDK_DEFAULT_ACCOUNT=$(aws sts get-caller-identity --query Account --output text)"
  );
}

new NnmctsPipelineStack(app, "NnmctsPipelineStack", {
  env: { account, region },
  description: "NNMCTS cloud GPU training pipeline (S3 + GPU EC2)",
});
