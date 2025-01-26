# File name: composed_client.py
import requests

msg = "Multiple 2 by 3"
response = requests.post("http://127.0.0.1:8000/", json=msg)

print(response.text)