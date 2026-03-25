import pandas as pd
from locust import FastHttpUser, task, constant_throughput
import os
from dotenv import load_dotenv

load_dotenv()
prompts = pd.read_csv("prompts.csv")
responses = pd.read_csv("responses.csv")

ENTITY_TYPES = [
    "NAME",
    "ADDRESS",
]


class MLServiceUser(FastHttpUser):
    host = os.getenv("GLINER_HOST", "http://localhost:8000")
    wait_time = constant_throughput(5)
    
    @task
    def predict_prompt(self):
        row = prompts.sample(n=1).iloc[0]
        prompt_text = row['user_msg']
        self.client.post(
            "/predict",
            json={
                "text": prompt_text,
            }
        )

    @task
    def predict_response(self):
        row = responses.sample(n=1).iloc[0]
        response_text = row['assistant_msg']
        self.client.post(
            "/predict",
            json={
                "text": response_text,
            }
        )
