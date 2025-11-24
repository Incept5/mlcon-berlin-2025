import time, ollama, re

def time_execution(func):
    def wrapper(i, task):
        start = time.time()
        result = func(task)
        elapsed = (time.time() - start) * 1000
        print(f"Task {i+1}: {task}:\n\t\t{result['response']} ({elapsed:.2f} ms)\n")
        return result
    return wrapper

client = ollama.Client()
run_query = time_execution(lambda p: client.generate(model="qwen3-vl:8b-instruct", prompt=p, think=False,  # Optional :30b-a3b-q8_0
                          system="Only provide the answer, no explanation"))

run_query(-1, "") # Initialise model so as not to screw up the first timing

tasks = [
    "Which British king abdicated in the 1936",
    "Where was Salvador Dali born?",
    "What is Einstein's most famous equation?",
    "In which country is Krakatoa?",
    "Write a single line in Java to print 'Hello World'",
    "Write a single line command for Linux to count the number of 20-letter words in the dictionary",
    "What is the derivative of 6x^2-5x+12?"
]

for i, task in enumerate(tasks):
    run_query(i, task)