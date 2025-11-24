import requests
import logging

logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)
requests_log = logging.getLogger("urllib3")
requests_log.setLevel(logging.DEBUG)
requests_log.propagate = True

try: # for Python 3
    from http.client import HTTPConnection
except ImportError:
    from httplib import HTTPConnection
HTTPConnection.debuglevel = 1

response = requests.post(
    "http://localhost:11434/api/generate",
    json={"model": "qwen3:4b", "prompt": "Hello",
          "stream": False, "Think": False }
)

data = response.json()
print(data["response"])
