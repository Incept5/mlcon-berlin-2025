from ollama import generate

prompt = """from ollama import generate

def call_llm(prompt):
# User a temperature of 0.7 and context of 4096
"""

suffix = """
    return result
    
def main():
  reply = call_llm("why is the sky blue?")
  print(reply)
"""

response = generate(
  model='llama3.2',
  prompt=prompt,
  suffix=suffix,
  options={
    'num_predict': 512,
    'temperature': 0,
    'top_p': 0.9,
    'stop': ['<EOT>'],
  },
)

print(response['response'])