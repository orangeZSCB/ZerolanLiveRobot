import os.path
from typing import Generator

import requests
from typeguard import typechecked
from zerolan.data.pipeline.asr import ASRQuery, ASRPrediction, ASRStreamQuery

from common.io.api import save_audio
from common.io.file_type import AudioFileType


class WhisperASRPipeline:

    def __init__(self, api_key: str, api_url: str, model: str = "whisper-1",
                 language: str | None = None, prompt: str | None = None,
                 temperature: float = 0.0, response_format: str = "json"):
        """
        Initialize Whisper ASR Pipeline.
        :param api_key: The API key for OpenAI/Whisper ASR service.
        :param api_url: The API URL for Whisper ASR service.
        :param model: The model ID to use. Currently only whisper-1 is available.
        :param language: The language of the input audio. ISO-639-1 format. Optional but improves accuracy.
        :param prompt: Optional text to guide the model's style or continue a previous audio segment.
        :param temperature: Sampling temperature between 0 and 1. Higher values make output more random.
        :param response_format: The format of the transcript output. Options: json, text, srt, verbose_json, vtt
        """
        self._api_key = api_key
        self._api_url = api_url
        self._model = model
        self._language = language
        self._prompt = prompt
        self._temperature = temperature
        self._response_format = response_format

    @typechecked
    def predict(self, query: ASRQuery) -> ASRPrediction:
        """
        Transcribe audio using Whisper API.
        :param query: ASR query containing audio path and metadata.
        :return: ASR prediction with transcript.
        """
        assert os.path.exists(query.audio_path), f"{query.audio_path} does not exist!"
        assert self._api_key is not None and self._api_key != "", "API key must be provided!"

        # Prepare multipart/form-data
        data = {
            'model': self._model,
        }
        
        # Add optional parameters
        if self._language is not None:
            data['language'] = self._language
        if self._prompt is not None:
            data['prompt'] = self._prompt
        if self._temperature != 0.0:
            data['temperature'] = str(self._temperature)
        if self._response_format != "json":
            data['response_format'] = self._response_format
        
        headers = {
            'Authorization': f'Bearer {self._api_key}'
        }
        
        # Open file and prepare for multipart upload
        with open(query.audio_path, 'rb') as audio_file:
            files = {
                'file': (os.path.basename(query.audio_path), audio_file, self._get_content_type(query.media_type))
            }
            
            response = requests.post(
                url=self._api_url,
                files=files,
                data=data,
                headers=headers
            )
        
        response.raise_for_status()
        
        # Parse response based on format
        if self._response_format == "json" or self._response_format == "verbose_json":
            result = response.json()
            if isinstance(result, dict):
                # Standard JSON format returns {"text": "..."}
                transcript = result.get('text', '')
                if not transcript:
                    # Fallback: try to get transcript from any key
                    transcript = str(result)
            else:
                transcript = str(result)
        elif self._response_format == "text":
            transcript = response.text
        else:
            # For srt, vtt formats, return text content
            transcript = response.text
        
        return ASRPrediction(transcript=transcript)

    def stream_predict(self, query: ASRStreamQuery, chunk_size: int | None = None) -> Generator[
        ASRPrediction, None, None]:
        """
        Stream predict is not directly supported by Whisper API.
        We convert stream query to regular query by saving audio to temp file.
        :param query: ASR stream query containing audio data.
        :param chunk_size: Not used for Whisper API.
        :return: Generator yielding ASR prediction.
        """
        audio_path = save_audio(query.audio_data, AudioFileType.WAV, prefix="asr")
        yield self.predict(ASRQuery(
            audio_path=str(audio_path),
            media_type=query.media_type,
            sample_rate=query.sample_rate,
            channels=query.channels,
        ))

    @staticmethod
    def _get_content_type(media_type: str) -> str:
        """
        Map media type to MIME content type.
        :param media_type: Audio file format (wav, mp3, etc.)
        :return: MIME content type
        """
        type_map = {
            "wav": "audio/wav",
            "mp3": "audio/mpeg",
            "mp4": "audio/mp4",
            "mpeg": "audio/mpeg",
            "mpga": "audio/mpeg",
            "m4a": "audio/mp4",
            "ogg": "audio/ogg",
            "webm": "audio/webm",
            "flac": "audio/flac"
        }
        return type_map.get(media_type.lower(), "audio/wav")
