import boto3 # AWS SDK (Software Development Kit) for Python to interact with AWS services
import json  # Used to formatting request and response payload 

# Initialize the Bedrock runtime client in the correct AWS region (setting up a connection)
bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-west-2"  
)

# Prompt that will be passed to the foundation model  
prompt = "Write a short story about a robot who wants to become a Skateboarder."

# Set test temperature value 
temperature = 0.9

# Build the input request and coverting the payload to JSON
body = json.dumps ({
    "inputText": prompt,
    "textGenerationConfig": {
        "maxTokenCount": 200,         # Limit the outpuut to 200 tokens 
        "temperature": temperature,   # Control randomness (0 = focused, 1 = creative)
        "topP": 1, 
        "stopSequences": []
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
print(f"\n--- Output (Temperature: {temperature}) ---\n")
print(json.dumps(response_body["results"][0]["outputText"], indent=2))


