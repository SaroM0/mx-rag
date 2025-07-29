# MX RAG Infrastructure

This directory contains the AWS CDK infrastructure code for deploying the MX RAG application.

## Architecture

The infrastructure consists of:

- VPC with 2 Availability Zones
- ECS Fargate Service running the application container
- Application Load Balancer for public access
- SNS Topic for scaling notifications
- AWS Secrets Manager for storing the OpenAI API key
- CloudWatch Log Group for container logs
- Auto-scaling configuration based on CPU and Memory utilization

## Prerequisites

1. AWS CLI installed and configured
2. Node.js 14.x or later
3. AWS CDK CLI installed (`npm install -g aws-cdk`)
4. Docker installed and running (for building the container image)

## Environment Setup

1. Install dependencies:
   ```bash
   npm install
   ```

2. Store your OpenAI API key in AWS Secrets Manager:
   ```bash
   aws secretsmanager create-secret \
     --name mx-rag/openai-api-key \
     --description "OpenAI API Key for MX RAG application" \
     --secret-string "your-api-key-here"
   ```

## Deployment

1. Bootstrap CDK (first time only):
   ```bash
   cdk bootstrap
   ```

2. Deploy the stack:
   ```bash
   npm run deploy
   ```

## Stack Outputs

After deployment, the following outputs will be available:

- `MxRagLoadBalancerDNS`: DNS name of the Application Load Balancer
- `MxRagScalingTopicArn`: ARN of the SNS topic for scaling notifications
- `MxRagServiceUrl`: URL of the RAG API service

## Development

- `npm run build`: Compile TypeScript code
- `npm run synth`: Synthesize CloudFormation template
- `npm run deploy`: Deploy the stack to AWS

## Security Considerations

- The OpenAI API key is stored securely in AWS Secrets Manager
- The Fargate service runs in private subnets
- The Application Load Balancer is the only public entry point
- Container logs are stored in CloudWatch Logs
- Auto-scaling is configured to handle load variations

## Cost Optimization

- NAT Gateway is limited to 1 instance to reduce costs
- Auto-scaling helps optimize resource usage
- Log retention is set to 1 week to manage storage costs
- Development environment uses different settings than production

## Monitoring and Maintenance

- Container insights are enabled for the ECS cluster
- CloudWatch Logs capture container output
- SNS notifications for scaling events
- Health checks are configured on the load balancer