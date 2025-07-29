#!/usr/bin/env node
import 'source-map-support/register';
import * as cdk from 'aws-cdk-lib';
import { MxRagStack } from '../lib/mx-rag-stack';

const app = new cdk.App();

// Get environment configuration
const environment = app.node.tryGetContext('environment') || process.env.ENVIRONMENT || 'development';
const notificationEmail = app.node.tryGetContext('notificationEmail') || process.env.NOTIFICATION_EMAIL;

new MxRagStack(app, 'MxRagStack', {
  env: {
    account: process.env.CDK_DEFAULT_ACCOUNT,
    region: process.env.CDK_DEFAULT_REGION || 'us-east-1'
  },
  environment,
  notificationEmail,
  description: `RAG API infrastructure stack (${environment} environment)`
});