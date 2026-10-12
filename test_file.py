from google import genai

# Initialize the client with your API key
client = genai.Client(api_key= "YOUR_API_KEY_HERE")

# Send a test prompt to the model
response = client.models.generate_content(
    model="gemini-3.8-flash",
    contents=(
        "Tell me a short, fun fact about Nigerian Jollof rice in one sentence."
    ),
)

# Print the result to your terminal
print("AI Response:")
print(response.text)