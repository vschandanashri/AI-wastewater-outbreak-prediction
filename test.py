import requests

data = {
    "virus_concentration": 80,
    "bacteria_level": 60,
    "temperature": 31,
    "ph_level": 7.1,
    "humidity": 68
}

response = requests.post(
    "http://127.0.0.1:5000/predict",
    json=data
)

print(response.json())
