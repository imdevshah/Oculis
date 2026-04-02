from agent.graph import app

inputs = {"input": "What is the revenue in 2025?"}
for output in app.stream(inputs):
    print(output)