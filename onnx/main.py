import litserve as ls
from gliner2_onnx import GLiNER2ONNXRuntime
from pydantic import BaseModel
import json
import logging
import os
from dotenv import load_dotenv
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PII_LABELS = ["person", "address", "email", "phone"]
SAFETY_LABELS = ["safe", "unsafe"]


class GuardRequest(BaseModel):
    text: str


class GLiNERGuardONNXAPI(ls.LitAPI):
    def setup(self, device):
        precision = os.getenv("ONNX_PRECISION", "fp16")
        _providers = os.getenv("ONNX_PROVIDERS", "")
        providers = [p.strip() for p in _providers.split(",") if p.strip()] or None
        _provider_options = os.getenv("ONNX_PROVIDER_OPTIONS", "")
        provider_options = json.loads(_provider_options) if _provider_options else None

        model_name = os.getenv("ONNX_MODEL_NAME", "hivetrace/gliner-guard-uniencoder-onnx")
        if os.path.isdir(model_name):
            logger.info("Loading ONNX model from local path: %s", model_name)
            self.runtime = GLiNER2ONNXRuntime(
                model_name,
                precision=precision,
                providers=providers,
                provider_options=provider_options,
            )
        else:
            logger.info("Loading ONNX model from Hugging Face repo: %s", model_name)
            self.runtime = GLiNER2ONNXRuntime.from_pretrained(
                model_name,
                precision=precision,
                providers=providers,
                provider_options=provider_options,
            )
        self.schema = (
            self.runtime.create_schema()
            .entities(entity_types=PII_LABELS, threshold=0.5)
            .classification(task="safety", labels=SAFETY_LABELS)
        )
        logger.info("device=%s  precision=%s  max_batch_size=%d", device, precision, self.max_batch_size)

    def decode_request(self, request: GuardRequest):
        return request.text

    def batch(self, inputs):
        return inputs

    def predict(self, batch):
        logger.info("batch_size=%d", len(batch))
        return self.runtime.extract_batch(batch, schema=self.schema)

    def unbatch(self, output):
        return output

    def encode_response(self, output):
        return output


if __name__ == "__main__":
    max_batch_size = int(os.getenv("MAX_BATCH_SIZE", 64))
    api = GLiNERGuardONNXAPI(max_batch_size=max_batch_size, batch_timeout=0.05)
    server = ls.LitServer(
        api,
        accelerator="auto",
        timeout=30,
        workers_per_device=int(os.getenv("NUM_WORKERS", 4)),
        fast_queue=True,
    )
    server.run(port=8000, generate_client_file=False)
