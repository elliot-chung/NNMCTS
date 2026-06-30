import * as cdk from "aws-cdk-lib";
import * as codebuild from "aws-cdk-lib/aws-codebuild";
import * as ec2 from "aws-cdk-lib/aws-ec2";
import * as iam from "aws-cdk-lib/aws-iam";
import * as logs from "aws-cdk-lib/aws-logs";
import * as s3 from "aws-cdk-lib/aws-s3";
import { Construct } from "constructs";
import * as fs from "fs";
import * as path from "path";
import { fileURLToPath } from "url";
import { smokeBuildSpec } from "./smoke-buildspec";

const currentDir = path.dirname(fileURLToPath(import.meta.url));
const gpuUserData = fs.readFileSync(path.join(currentDir, "../../cloud/gpu-train.sh"), "utf8");

export class NnmctsPipelineStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    const artifactsBucket = new s3.Bucket(this, "ArtifactsBucket", {
      bucketName: `nnmcts-artifacts-${this.account}-${this.region}`,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      encryption: s3.BucketEncryption.S3_MANAGED,
      enforceSSL: true,
      versioned: true,
      removalPolicy: cdk.RemovalPolicy.RETAIN,
      autoDeleteObjects: false,
    });

    const sourcePrefix = "source";
    const logGroup = new logs.LogGroup(this, "CodeBuildLogs", {
      logGroupName: "/nnmcts/codebuild",
      retention: logs.RetentionDays.ONE_MONTH,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });

    const codeBuildRole = new iam.Role(this, "CodeBuildRole", {
      assumedBy: new iam.ServicePrincipal("codebuild.amazonaws.com"),
      description: "Runs NNMCTS CPU smoke training jobs in CodeBuild",
    });

    artifactsBucket.grantReadWrite(codeBuildRole);
    logGroup.grantWrite(codeBuildRole);

    const smokeProject = new codebuild.Project(this, "SmokeTrainingProject", {
      projectName: "nnmcts-smoke-training",
      description: "End-to-end NNMCTS CPU smoke training (max 1 hour)",
      role: codeBuildRole,
      timeout: cdk.Duration.hours(1),
      queuedTimeout: cdk.Duration.minutes(30),
      environment: {
        buildImage: codebuild.LinuxBuildImage.STANDARD_7_0,
        computeType: codebuild.ComputeType.MEDIUM,
        privileged: false,
      },
      environmentVariables: {
        ARTIFACTS_BUCKET: {
          type: codebuild.BuildEnvironmentVariableType.PLAINTEXT,
          value: artifactsBucket.bucketName,
        },
      },
      logging: {
        cloudWatch: {
          logGroup,
          enabled: true,
        },
      },
      buildSpec: smokeBuildSpec,
    });

    const vpc = new ec2.Vpc(this, "GpuVpc", {
      maxAzs: 1,
      natGateways: 0,
      subnetConfiguration: [
        {
          name: "public",
          subnetType: ec2.SubnetType.PUBLIC,
          cidrMask: 24,
        },
      ],
    });

    const gpuSecurityGroup = new ec2.SecurityGroup(this, "GpuTrainingSecurityGroup", {
      vpc,
      description: "NNMCTS GPU training instances (egress only)",
      allowAllOutbound: true,
    });

    const gpuInstanceRole = new iam.Role(this, "GpuTrainingInstanceRole", {
      assumedBy: new iam.ServicePrincipal("ec2.amazonaws.com"),
      description: "NNMCTS GPU training EC2 instance role",
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName("AmazonSSMManagedInstanceCore"),
      ],
    });
    artifactsBucket.grantReadWrite(gpuInstanceRole);

    const gpuInstanceProfile = new iam.CfnInstanceProfile(this, "GpuTrainingInstanceProfile", {
      roles: [gpuInstanceRole.roleName],
    });

    const gpuAmi = ec2.MachineImage.genericLinux({
      "us-west-1": "ami-0b2f6fd4ed32fc52d",
    });

    const gpuLaunchTemplate = new ec2.CfnLaunchTemplate(this, "GpuTrainingLaunchTemplate", {
      launchTemplateName: "nnmcts-gpu-training",
      launchTemplateData: {
        imageId: gpuAmi.getImage(this).imageId,
        instanceType: "g4dn.xlarge",
        iamInstanceProfile: { arn: gpuInstanceProfile.attrArn },
        networkInterfaces: [
          {
            deviceIndex: 0,
            associatePublicIpAddress: true,
            groups: [gpuSecurityGroup.securityGroupId],
            subnetId: vpc.publicSubnets[0].subnetId,
          },
        ],
        blockDeviceMappings: [
          {
            deviceName: "/dev/xvda",
            ebs: {
              volumeSize: 50,
              volumeType: "gp3",
              deleteOnTermination: true,
            },
          },
        ],
        tagSpecifications: [
          {
            resourceType: "instance",
            tags: [{ key: "Name", value: "nnmcts-gpu-training" }],
          },
        ],
        metadataOptions: {
          httpTokens: "required",
          httpPutResponseHopLimit: 2,
          instanceMetadataTags: "enabled",
        },
        userData: cdk.Fn.base64(gpuUserData),
      },
    });

    new cdk.CfnOutput(this, "ArtifactsBucketName", {
      value: artifactsBucket.bucketName,
      description: "S3 bucket for training runs and source uploads",
    });

    new cdk.CfnOutput(this, "SourceUploadPrefix", {
      value: `s3://${artifactsBucket.bucketName}/${sourcePrefix}/`,
      description: "Upload zipped source code here before starting a build",
    });

    new cdk.CfnOutput(this, "CodeBuildProjectName", {
      value: smokeProject.projectName,
      description: "CodeBuild project for CPU smoke training runs",
    });

    new cdk.CfnOutput(this, "GpuLaunchTemplateName", {
      value: gpuLaunchTemplate.launchTemplateName!,
      description: "EC2 launch template for GPU training (g4dn.xlarge, 1h train / 90m instance cap, auto-shutdown)",
    });

    new cdk.CfnOutput(this, "LogGroupName", {
      value: logGroup.logGroupName,
      description: "CloudWatch log group for CodeBuild output",
    });
  }
}
