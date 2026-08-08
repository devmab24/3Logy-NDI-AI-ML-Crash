## AWS Account Setup Guide for SIRA Development
Smart Incident Report Analyzer (SIRA)
This guide helps you create and configure an AWS account so the SIRA project can progressively move into the AWS environment.
> **Important:** AWS is a real cloud environment. Some services can incur charges. Monitor usage and delete resources when they are no longer needed.

### What we are preparing for
   
Current workflow:
```text
Local Dataset → Data Cleaning → Feature Engineering → Model Training → Evaluation → Prediction
```

Target workflow:
```text
S3 → SageMaker → Model → SIRA Application
```

### Before we start
   
Prepare a working email address, phone number, accepted payment method, and computer with internet access.
Keep all AWS credentials private. Never share passwords, access keys, secret keys, or MFA codes.

### Create your AWS account
   
Go to the official AWS website and select `Create an AWS Account`.
Complete the registration, email/phone verification, payment information, and support-plan selection. Follow the guidance and do not purchase additional support unless instructed.

### Sign In
   
Open the AWS Management Console and sign in after registration.

### AWS Regions
   
AWS resources are created in geographic Regions. Use the Region specified by your facilitator. Do not create resources in multiple Regions unnecessarily.

### Secure the Root User
The root user has unrestricted account access. Enable MFA and do not use the root user for everyday development. Do not create root access keys.

### IAM and least privilege
   
AWS Identity and Access Management (IAM) controls who can access resources and what they can do.
Use appropriate IAM identities for development and follow the Principle of Least Privilege: grant only the permissions required for the task.

### AWS CLI
Install the AWS CLI from the official documentation:
https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html
Verify:
```bash
aws --version
```
Configure the CLI using the authentication method provided by your facilitator. Never hard-code credentials in Python or commit them to GitHub.
Verify the active identity:
```bash
aws sts get-caller-identity
```

### Install Boto3
Boto3 is the AWS SDK for Python and will allow SIRA to interact with AWS services programmatically.
```bash
pip install boto3
```
Verify:
```bash
pip show boto3
```

### Prepare for Amazon S3
S3 will become the cloud storage layer for SIRA:
```text
sira-data/
├── raw/
│   └── incident_reports_1000.csv
├── processed/
│   └── incident_reports_clean.csv
└── models/
```
Keep buckets private. Do not upload credentials, passwords, personal information, or confidential data.

### Billing and Cost Awareness
Review AWS Billing and Cost Management. Become familiar with the billing dashboard, Cost Explorer, budgets, and alerts. Be especially careful with continuously running resources such as SageMaker endpoints, EC2 instances, and databases.
Before creating a resource, ask:
What am I creating?
Why am I creating it?
What could it cost?
When should it be deleted?

### Do Not Create SageMaker Endpoints Yet
For now, complete account and security setup only. Our progression will be:
```text
AWS Account → IAM/Security → AWS CLI/Boto3 → S3 → SageMaker → SIRA Deployment
```

### Verification Checklist
[ ] AWS account created
[ ] Email and phone verified
[ ] Root MFA enabled
[ ] Development IAM identity configured
[ ] AWS CLI installed
[ ] `aws --version` works
[ ] `aws sts get-caller-identity` works
[ ] Boto3 installed
[ ] Billing dashboard reviewed
[ ] Cost monitoring/budget reviewed
[ ] No credentials committed to GitHub

### What Comes Next
After setup, we will move the SIRA dataset into Amazon S3, introduce SageMaker, and progressively connect the existing ML pipeline to AWS. These practical activities will also reinforce concepts relevant to the AWS Certified AI Practitioner examination.
Security Reminder
Never send your AWS password, access keys, secret access keys, MFA codes, or other credentials to instructors, classmates, WhatsApp, GitHub, or ChatGPT. If credentials are exposed, revoke/rotate them immediately.
Official AWS Resources
AWS: https://aws.amazon.com/
AWS Console: https://console.aws.amazon.com/
AWS CLI: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html
IAM: https://docs.aws.amazon.com/iam/
S3: https://docs.aws.amazon.com/s3/
SageMaker: https://docs.aws.amazon.com/sagemaker/
AWS Certified AI Practitioner: https://aws.amazon.com/certification/certified-ai-practitioner/
