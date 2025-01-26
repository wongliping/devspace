from starlette.requests import Request

import ray
from ray import serve

from transformers import pipeline

# decorator converts Translator from a Python class into a Ray Serve Deployment object
@serve.deployment(num_replicas=2, ray_actor_options={"num_cpus": 0.2, "num_gpus": 0})
class Translator:
    def __init__(self):
        # Load model
        self.model = pipeline("translation_en_to_fr", model="t5-small")

    def translate(self, text: str) -> str:
        # Run inference
        model_output = self.model(text)

        # Post-process output to return only the translation text
        translation = model_output[0]["translation_text"]

        return translation

    async def __call__(self, http_request: Request) -> str:
        """
        Handles incoming HTTP requests and translates the provided English text.

        Deployment receive Starlette HTTP request objects
        Return value is sent back in the HTTO response body

        Processes incoming HTTP request by reading its JSON data and forwarding it to the translate method.

        Args:
            http_request (Request): The incoming HTTP request containing JSON data.

        Returns:
            str: The translated text.
        """
        english_text: str = await http_request.json()
        return self.translate(english_text)


# Bind Translator deployment to arguments that will be passed into its constructor
# Defines a Ray Serve application (which can consist of multiple deployments)
translator_app = Translator.bind()