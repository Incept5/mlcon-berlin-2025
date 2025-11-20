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
run_query = time_execution(lambda p: client.generate(model="qwen3:4b-instruct", prompt=p,
                          system="You are a clever assistant, provide simple answers with no explanation in English only"))

run_query(-1, "") # Initialise model so as not to screw up the first timing

tasks = [
    "Which British king abdicated in the 1936",
    "Wann war das erste Oktoberfest?",
    "¿Dónde nació Salvador Dalí?",
    "Lequel des deux est le plus ancien, la Tour Eiffel ou le Sacré-Cœur ?",
    "什么是爱因斯坦最著名的方程式？",
    "في أي بلد يقع كراكاتوا؟",
    "Write a single line in Java to print 'Hello World'",
    "Write a single line command for Linux to count the number for 20-letter words in the dictionary",
    "What is the derivative of 6x^2-5x+12?"
]

for i, task in enumerate(tasks):
    run_query(i, task)