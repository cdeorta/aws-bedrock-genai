import boto3
import json

# Initialize the AWS Bedrock runtime client 
session = boto3.Session(region_name="us-west-2")
bedrock_runtime = session.client("bedrock-runtime")

# Model ID for Cohere Command R
model_id = "cohere.command-r-v1:0"

# Example demonstrating multi-turn chat capability
# Chat history lets the model remember previous interactions and maintain context
chat_history = [
    {"role": "USER", "message": "What is cloud computing?"},
    {"role": "CHATBOT", "message": "Cloud computing is the delivery of computing services like servers, storage, databases, networking, software, and more over the Internet ('the cloud')."},
    {"role": "USER", "message": "Can you explain that in simpler terms for a child?"}
]

# Message (latest question from the user)
message = "Can you now explain it with a fun analogy or story?"

# Prepare request with penalties to discourage repetition of common terms
body = json.dumps({
    "message": message,
    "chat_history": chat_history,
    "max_tokens": 200,              # limit output length
    "temperature": 0.8,             # creativity balance
    "p": 0.9,                       # Top-p sampling for coherent responses
    "frequency_penalty": 0.3,       # penalizes repeating the same words frequently
    # "presence_penalty": 0.5         # Command R model validation internally rejects certain penalty combinations
})

# Invoke Cohere Command R model
response = bedrock_runtime.invoke_model(
    modelId=model_id,
    body=body,
    contentType="application/json",
    accept="application/json"
)

# Parse and display response
response_body = json.loads(response['body'].read())
answer = response_body.get('text', '')

print("Cohere Command R Multi-Turn Response:")
print(answer)

"""
Cohere Command R Multi-Turn Response:
Certainly! Imagine you have a big, powerful computer that's capable of doing all sorts of amazing things. But it's so big and heavy that it lives somewhere far away, in a special place called the Cloud.
This computer is like a magical friend who can help you with lots of fun and exciting tasks. You can ask it to play games with you, store your pictures, help you with your homework, or even turn on your favorite cartoon! It's always there and ready to assist you.
You access this super computer using your own device, whether it's a phone, tablet, or regular computer. It's like having a special key that unlocks this huge computer's powers, but instead of being a physical key, it's an internet connection!
So, every time you watch a video, play a game, or use a cool app, you're actually asking your friendly computer in the Cloud to help you out. The Cloud makes it possible for you to do
"""
