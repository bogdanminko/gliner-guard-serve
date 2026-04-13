"""
Locust test for GLiNER Guard via vLLM (vllm-factory deberta_gliner2 plugin).

Uses the same test data (prompts.csv / responses.csv) as the LitServe tests,
but targets vLLM's /pooling endpoint with the deberta_gliner2 schema format.

Usage:
    GLINER_HOST=http://localhost:8000 \
    uv run locust -f test-gliner-vllm.py -u 100 -r 1 --run-time 15m --csv=vllm-stats
"""

import os

import pandas as pd
from dotenv import load_dotenv
from locust import FastHttpUser, constant_throughput, task

load_dotenv()
prompts = pd.read_csv("prompts.csv")
responses = pd.read_csv("responses.csv")

VLLM_MODEL = os.getenv("VLLM_MODEL", "/tmp/gliner-guard-uni-vllm")

PII_LABELS = ["person", "address", "email", "phone"]
SAFETY_LABELS = ["safe", "unsafe"]

GLINER2_SCHEMA = {
    "entities": PII_LABELS,
    "classifications": [
        {"task": "safety", "labels": SAFETY_LABELS}
    ],
}


class VLLMGlinerUser(FastHttpUser):
    host = os.getenv("GLINER_HOST", "http://localhost:8000")
    wait_time = constant_throughput(5)

    @task
    def predict_prompt(self):
        row = prompts.sample(n=1).iloc[0]
        self.client.post(
            "/pooling",
            json={
                "model": VLLM_MODEL,
                "task": "plugin",
                "data": {
                    "text": row["user_msg"],
                    "schema": GLINER2_SCHEMA,
                    "threshold": 0.4,
                },
            },
        )

    @task
    def predict_response(self):
        row = responses.sample(n=1).iloc[0]
        self.client.post(
            "/pooling",
            json={
                "model": VLLM_MODEL,
                "task": "plugin",
                "data": {
                    "text": row["assistant_msg"],
                    "schema": GLINER2_SCHEMA,
                    "threshold": 0.4,
                },
            },
        )
