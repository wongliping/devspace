# File name: serve_quickstart_composed.py
from starlette.requests import Request

import ray
from ray import serve
from ray.serve.handle import DeploymentHandle

from transformers import pipeline


@serve.deployment
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


@serve.deployment
class Summarizer:
    """
    A class used to summarize text using a pre-trained summarization model.

    Attributes
    ----------
    translator : DeploymentHandle
        A handle to a translation deployment service.
    model : pipeline
        A pre-trained summarization model from the Hugging Face Transformers library.

    Methods
    -------
    summarize(text: str) -> str
        Summarizes the given text and returns the summary.
    
    async __call__(http_request: Request) -> str
        Asynchronously handles an HTTP request, summarizes the text, and translates the summary.
    """
    def __init__(self, translator: DeploymentHandle):
        """
        Takes in a handle to the Translator as one of its constructor arguments,
        so it can forward summarized texts to the Translator deployment
        """
        self.translator = translator

        # Load model.
        self.model = pipeline("summarization", model="t5-small")

    def summarize(self, text: str) -> str:
        # Run inference
        model_output = self.model(text, min_length=5, max_length=15)

        # Post-process output to return only the summary text
        summary = model_output[0]["summary_text"]

        return summary

    async def __call__(self, http_request: Request) -> str:
        english_text: str = await http_request.json()
        summary = self.summarize(english_text)

        # self.translator.translate.remote(summary) issues an asynchronous call to the Translator’s translate method 
        # and returns a DeploymentResponse object immediately
        # calling await on the response waits for the remote method call to execute and returns its return value
        # response could also be passed directly to another DeploymentHandle call
        translation = await self.translator.translate.remote(summary)
        return translation


app = Summarizer.bind(Translator.bind())