import boto3 # AWS SDK (Software Development Kit) for Python to interact with AWS services
import json  # Used to formatting request and response payload 

# Initialize the Bedrock runtime client in the correct AWS region (setting up a connection)
bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-west-2"  
)

# Set the prompt for testing Top P
prompt = "Describe a sunset in one sentence."

# Keep temp steady to isolate topP impact (keep some creativity)
temperature = 0.7

# Modify Top P value here to experiment: test 1.0, 0.5, and 0.1
top_p_value = 0.1 

# Build the input request and coverting the payload to JSON
body = json.dumps ({
    "inputText": prompt,
    "textGenerationConfig": {
        "maxTokenCount": 50,         
        "temperature": temperature, 
        "topP": top_p_value,         # Parameter thats being tested                     
    }
})

# Call the Titan text model via Bedrock 
response = bedrock_runtime.invoke_model(
    body=body,
    modelId="amazon.titan-text-express-v1",  
    accept="application/json",
    contentType="application/json"
)


# Decode and print the model's response
response_body = json.loads(response["body"].read())
print(json.dumps(response_body, indent=2))


