import boto3
import json

# Initialize the Bedrock runtime client
session = boto3.Session(region_name="us-west-2")
bedrock_runtime = session.client("bedrock-runtime")

# Model ID for Meta Llama 3 (specific: Llama 3 70B Instruct)
model_id = "meta.llama3-70b-instruct-v1:0"

# Meta Llama 3 prompt follows a structured instruction format
prompt = """### Instruction:
Summarize the following text in four concise sentence:

Amazon Web Services (AWS) provides on-demand cloud computing platforms and APIs to individuals, companies, and governments on a metered pay-as-you-go basis. AWS offers a wide range of cloud-based products including computing, storage, databases, analytics, networking, and developer tools.

### Response:"""

# Define inference parameters
body = json.dumps({
    "prompt": prompt,
    "max_gen_len": 100,  # length of generated output
    "temperature": 0.5,  # Moderate randomness 
    "top_p": 0.7        # Probability threshold for token selection
})

# Invoke the Meta Llama 3 model
response = bedrock_runtime.invoke_model(
    modelId=model_id,
    body=body,
    contentType="application/json",
    accept="application/json"
)

# Decode and print response from Meta Llama 3
response_body = json.loads(response['body'].read())
output = response_body.get('generation', '')

print("Meta Llama 3 Summary:")
print(output)

"""
Meta Llama 3 Summary:
Here is a summary of the text in four concise sentences:
Amazon Web Services (AWS) offers cloud computing platforms and APIs on a pay-as-you-go basis. The service is available to individuals, companies, and governments.
 AWS provides a broad range of cloud-based products, including computing, storage, and databases. 
These products also include analytics, networking, and developer tools.
"""