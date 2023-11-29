AWS with Python – Single Account

WARNING: When using AWS with Python, you need to establish a root user, which serves as an administrator account. AWS then recommends establishing a second, dependent account(called IAM), which the root user assigns rights to. In this, This tutorial will let you use AWS only using a single root user account, bypassing the setup of the dependent account. Please proceed with caution and ensure that you understand the security implications of using the root user for AWS tasks. It's recommended to follow best practices by using IAM users or roles for improved security and accountability.

1. **Create an AWS Account**: If you haven't already, sign up for an AWS account through the AWS website (<https://aws.amazon.com/>).
1. **Install the AWS CLI:** Install the AWS Command Line Interface (CLI) on your local machine. You can download and install it from here: <https://aws.amazon.com/cli/>.
1. E**stablish your AWS access key and secret key**.
   1. Log in to the AWS Management Console: Go to the AWS Management Console at <https://aws.amazon.com/> and sign in using your AWS account credentials.
   1. Access the IAM Console: Once you're logged in, open the AWS Identity and Access Management (IAM) console. You can do this by clicking on the "Services" dropdown in the top left corner and selecting "IAM" under the "Security, Identity, & Compliance" section.
   1. Click "My security credentials", then create an access key.
1. **Configure AWS CLI with Root User Credentials:** Run the following commands in terminal to configure the AWS CLI with your root user credentials. Replace YOUR\_ACCESS\_KEY, YOUR\_SECRET\_KEY, and YOUR\_REGION with your actual AWS root user access key and secret key.
   1. For the region, there is a dropdown in the top right with different regions. 

I used us-east-2, since I am in Illinois.

1. aws configure
1. AWS Access Key ID [None]: YOUR\_ACCESS\_KEY
1. AWS Secret Access Key [None]: YOUR\_SECRET\_KEY
1. Default region name [None]: YOUR\_REGION
1. Default output format [None]: json

Now you are done setting up AWS for use with Python.
