# Gata Finetune

Gata Finetune is part of the [Gata Router](https://gata.works) project - an open source AI-powered 
support ticket triage/routing platform. This component handles the finetuning of a [BERT 
transformer model](https://huggingface.co/docs/transformers/en/model_doc/bert) 
(`bert-base-uncased`) to improve ticket classification accuracy.

## Input Data Requirements

The finetune script expects input data in [JSON Line (aka JSONL)](https://jsonlines.org) format 
with your training and evaluation datasets in separate files. Each file needs to be named 
`data.json` and placed in its own directory. Check out the [schema](./training-data.schema.json) 
for details on how to structure each row.

### Data Format

Each line in your `data.json` file should be a JSON object with two fields:

```json
{"label": 0, "text": "password reset request for user account"}
{"label": 1, "text": "billing inquiry about monthly invoice"}
{"label": 2, "text": "software installation help needed urgently"}
```

- `label`: An integer (0 or greater) representing the ticket destination/category
- `text`: Lowercase text containing only letters, numbers, spaces, and basic punctuation

### Directory Structure

Organize your data files like this:

```
data/
  train/
    data.json    # Training dataset
  test/
    data.json    # Evaluation dataset
```

For effective finetuning, we've found reasonably well balanced datasets with around 5,000 examples
work well. You can start with fewer if that's what you have available. The more representative
examples you have, the better the model will perform.

**Note:** For production use, data should be prepared using the [Gata Data Prep 
job](https://github.com/gata-router/gata-data-prep), which handles data cleaning, validation, and 
formatting to ensure optimal training results.

## Environment Setup

The finetune image is designed to run on AWS SageMaker. You can run it locally, but you're on 
your own. The finetune script and container are built to run on AWS SageMaker.

The build process is optimized for runtime performance. It's deliberately slow to build, which 
means it starts and runs faster when you actually need it.

## Deployment

You'll need to build the Docker image and deploy it to AWS ECR for use with SageMaker.

### Setting Up with Terraform

The [Gata Terraform module](https://github.com/gata-router/terraform-aws-gata) sets up the necessary infrastructure:

- ECR repository for your finetune images
- IAM role for GitHub Actions to assume via OIDC
- SSM Parameter to track release versions

### GitHub Actions Workflow

To automate building and pushing images to ECR:

1. Copy the workflow template:
   ```sh
   cp .github/workflows/release.yaml.txt .github/workflows/release.yaml
   ```

2. Configure the following repository variables and secrets in your GitHub repository settings:
   - Variables: AWS region, ECR repository name, etc.
   - Secrets: Any sensitive configuration values

The workflow will automatically build and push images to your ECR repository when you tag a release using the YYYYMMDDHH format.

## Running Locally for Testing

To test your changes locally you first need to build the docker image:

```sh
docker build -t finetune .
```

It takes 5-10 minutes to build the image, depending on your network speed. It pulls down a lot of 
content from the internet.

Once the image is built you can run it with:

```sh
docker run --rm \
 -v $(pwd)/data/input/data/train:/data/train \
 -v $(pwd)/data/input/data/test:/data/test \
 -v $(pwd)/data/output/data:/data/output \
 -v $(pwd)/data/model:/data/model \
 -e "PYTORCH_ENABLE_MPS_FALLBACK=1" \
 -e "SM_OUTPUT_DATA_DIR=/data/output" \
 -e "SM_MODEL_DIR=/data/model" \
 -e "SM_NUM_GPUS=0" \
 -e "SM_CHANNEL_TRAIN=/data/train" \
 -e "SM_CHANNEL_TEST=/data/test" \
 -e "TOKENIZERS_PARALLELISM=false" \
 finetune:latest train
```

This command sets up the SageMaker like environment variables locally and starts the training 
process. Expect the finetuning to take anywhere from a few minutes to many hours depending on 
your hardware and dataset size.

## Hardware Requirements

The finetune job performs well on an AWS `ml.g6.xlarge` instance (4 vCPU, 16GB RAM, with GPU). 
It has been tested with data containing up to 100 000 rows. To control costs, consider running 
training jobs on spot instances.

For local testing, you'll want at least 16GB of RAM and a decent CPU. GPU acceleration will 
dramatically speed up the process but isn't strictly required for testing.

## After Finetuning

When the finetuning completes, you'll find your model saved as a tarball in the directory 
specified by `SM_MODEL_DIR`. The model can be loaded directly by the Gata Router system or 
used independently with the Hugging Face Transformers library.

Typical evaluation metrics to look for include accuracy, F1 score, and confusion matrices for your 
ticket classifications. These metrics will be output to the console during training and saved to 
the output directory.

## Getting Help

If you encounter issues specific to the finetuning process, please open an issue in this 
repository. For questions about how this component integrates with the broader Gata Router 
platform, check out the main project documentation at https://gata.works.