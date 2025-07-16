import boto3
import json

# Initialize the Bedrock runtime client (assumes credentials and region are configured)
bedrock_runtime = boto3.client(service_name="bedrock-runtime")

# Claude 3 Sonnet model ID
model_id = "anthropic.claude-3-sonnet-20240229-v1:0"

# Define the conversation history (multi-turn)
messages = [
    # turn 1: User greeting
    {"role": "user", "content": "hello claude, what can you help me with today?"},
    
    # turn 2: Claude's introducting 
    {"role": "assistant", "content": "Hi! I can help with learning, productivity, summaries, and more. What do you need?"},
   
   # turn 3: User requests explanation on specific topic 
    {"role": "user", "content": "Can you explain what machine learning is in a simple way?"}
]

# System prompt (sets Claude's behavior and tone)
system_prompt = "You are Claude, a patient AI tutor for beginners in tech. Keep answers clear and use analogies."

# Define inference parameters
body = {
    "anthropic_version": "bedrock-2023-05-31",  # required for Claude models
    "system": system_prompt,                    # Claude will follow this instruction closely
    "messages": messages,                       # our conversation history
    "max_tokens": 300,                          # response length control
    "temperature": 0.7,                         # balanced creativity
    "top_p": 0.9                                # controls diversity
}

# Convert Python dict to JSON string
json_body = json.dumps(body)

# Call the Claude 3 Sonnet model
response = bedrock_runtime.invoke_model(
    modelId=model_id,
    body=json_body,
    contentType="application/json",
    accept="application/json"
)

# Read and decode response
response_body = json.loads(response['body'].read())

# Extract and display Claude's response clearly
claude_response = response_body.get('content', [])
if claude_response:
    print("Claude 3 Sonnet Response:")
    for message in claude_response:
        if message['type'] == 'text':
            print(message['text'])
else:
    print("No response received from Claude.")