import os
from fastapi import FastAPI, Request
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import json
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.images_response import ImagesResponse
from rcs import (
    Action,
    Card,
    InboundActionMessage,
    InboundLocationMessage,
    InboundMediaMessage,
    InboundTextMessage,
    Pinnacle,
    SendRcsResponse,
)
from dotenv import load_dotenv
from openai import OpenAI
import requests
from io import BytesIO
from PIL import Image
from PIL.Image import Image as PILImage
import PIL
from pillow_heif import register_heif_opener
import base64
import fal_client

load_dotenv()

KEY: str | None = os.getenv("PINNACLE_API_KEY")
if not KEY:
    raise ValueError("No key provided")

client = Pinnacle(api_key=KEY)
openai_client = OpenAI()

app = FastAPI()


class MessageHandler:
    def __init__(self, signing_secret: str):
        self.signing_secret = signing_secret

    def verify_request(self, received_secret: str) -> bool:
        print(
            f"Comparing secrets - Received: {received_secret}, Expected: {self.signing_secret}"
        )
        return received_secret == self.signing_secret

    def handle_message(self, message_data: dict, received_secret: str) -> None:

        # First verify the request
        if not self.verify_request(received_secret):
            raise ValueError("Invalid signing secret")

        # Parse the message using Pinnacle
        message: (
            InboundActionMessage
            | InboundTextMessage
            | InboundLocationMessage
            | InboundMediaMessage
        ) = Pinnacle.parse_inbound_message(message_data)
        print(f"Received message: {message}")
        print(f"Message type: {message.message_type}")

        # Handle different message types
        if message.message_type == "text":
            self._handle_text_message(message)
        elif message.message_type == "media":
            self._handle_media_message(message)
        elif message.message_type == "action":
            self._handle_action_message(message)
        elif message.message_type == "location":
            self._handle_location_message(message)
        else:
            print(f"Unhandled message type: {message.message_type}")

    def _handle_text_message(self, message: InboundTextMessage):
        print(f"Processing text message: {message.text}")
        if message.text == "Hello!":
            print("Hello, World!")
            response: SendRcsResponse = client.send.rcs(
                to=message.from_, from_="test", text="Hello, World!"
            )  # from_ can be your agent id as well if you're not using a test agent
            print(f"Response: {response.message}, {response.message_id}")

    def _handle_media_message(self, message: InboundMediaMessage):
        print(f"Processing image message: {message}")
        print(f"Media type: {message.media_urls[0].url}")

        try:
            # Download the image from the URL
            response: requests.Response = requests.get(url=message.media_urls[0].url)
            response.raise_for_status()

            try:
                # Check if it's a HEIC file by looking at the first few bytes
                file_header = response.content[:12]
                is_heic = b"ftypheic" in file_header or b"ftypmif1" in file_header

                if is_heic:
                    print("Detected HEIC format")

                # Register HEIF opener before attempting to open images
                register_heif_opener()

                # Attempt to open the image
                print(
                    f"Attempting to open image, content length: {len(response.content)}"
                )
                image: PILImage = Image.open(BytesIO(response.content))
                print(
                    f"Image opened successfully. Mode: {image.mode}, Size: {image.size}"
                )

                # Convert to RGB if necessary (HEIC images might be in RGBA)
                if image.mode != "RGB":
                    print(f"Converting image from {image.mode} to RGB")
                    image = image.convert("RGB")

                # Resize the image while maintaining aspect ratio
                max_size = 1024  # Maximum dimension
                ratio = min(max_size / image.size[0], max_size / image.size[1])
                new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
                print(f"Resized image to: {new_size}")

                # Create a BytesIO object to store the PNG
                print("Creating PNG buffer")
                png_buffer = BytesIO()
                # Use JPEG format instead of PNG for smaller file size
                image.save(png_buffer, format="JPEG", quality=85)
                png_buffer.seek(0)
                print(f"Image buffer size: {len(png_buffer.getvalue())}")

                # Send the original image back as a card
                rcs_response: SendRcsResponse = client.send.rcs(
                    to=message.from_,
                    from_="test",
                    text="Creating your polaroid...",
                )

                astica_api_key = os.getenv("ASTICA_API_KEY")
                if not astica_api_key:
                    raise ValueError(
                        "ASTICA_API_KEY not found in environment variables"
                    )

                astica_payload = {
                    "tkn": astica_api_key,
                    "modelVersion": "2.5_full",
                    "visionParams": "gpt_detailed,describe,objects,faces",
                    "input": f"data:image/png;base64,{base64.b64encode(png_buffer.getvalue()).decode('utf-8')}",
                    "gpt_prompt": "Describe the image and give an overview of the scene. Pay attention to certain things the subject is wearing and what they're doing.",
                    "prompt_length": 250,
                }

                try:
                    response = requests.post(
                        "https://vision.astica.ai/describe",
                        json=astica_payload,
                        timeout=25,
                        headers={"Content-Type": "application/json"},
                    )
                    
                    # Check status code before parsing JSON
                    response.raise_for_status()
                    result = response.json()

                    if result["status"] == "success":
                        description = result.get(
                            "caption_GPTS", "No description available."
                        )
                        client.send.rcs(
                            to=message.from_, from_="test", text="Almost done..."
                        )

                        try:
                            completion: ChatCompletion = (
                                openai_client.chat.completions.create(
                                    model="gpt-4",
                                    messages=[
                                        {
                                            "role": "system",
                                            "content": "You are a creative character designer. Transform the description of a image into a description of a raccoon in that same scenario that maintains the key characteristics and style of the original image. Personify the racccoon with the clothes or appearance of the subject.",
                                        },
                                        {
                                            "role": "user",
                                            "content": f"Original description:\n{description}",
                                        },
                                    ],
                                    max_tokens=250,
                                    temperature=0.7,
                                )
                            )

                            enhanced_description = completion.choices[0].message.content
                            print(f"\n\n***Enhanced description***\n: {enhanced_description}\n\n")

                            def on_queue_update(update):
                                if isinstance(update, fal_client.InProgress):
                                    for log in update.logs:
                                        print(log["message"])

                            result = fal_client.subscribe(
                                "fal-ai/flux/dev",
                                arguments={
                                    "prompt": f"Polaroid photo that is a square or rectangular image with a glossy surface, surrounded by a thick white border, wider at the bottom. At the bottom \"PINNACLE 12/2024\" is scribbled in black sharpie.\nThe image often has a soft focus, warm tones, and a vintage feel. Photorealistic Style. If the raccoon is a girl, add a pink bow on her. The subject of the image is a raccoon who looks like this person:\nPERSON DESCRIPTION:\n{enhanced_description}"
                                },
                                with_logs=True,
                                on_queue_update=on_queue_update,
                            )
                            print(result)
                            print(f"Description: {description}")
                            image_url = result["images"][0]["url"]
                            client.send.rcs(
                                to=message.from_,
                                from_="test",
                                text="It's ready!",
                            )
                            print("Media url:", message.media_urls[0].url)
                            cards = [Card(title="After", media_url=image_url)]
                            
                            # Only add the "Before" card if it's not a HEIC image
                            if not is_heic:
                                cards.insert(0, Card(
                                    title="Before",
                                    media_url=message.media_urls[0].url,
                                ))
                                
                            res: SendRcsResponse = client.send.rcs(
                                to=message.from_,
                                from_="test",
                                cards=cards,
                            )
                            print("res:", res)

                        except Exception as e:
                            print(f"Error with OpenAI GPT-4: {e}")
                            client.send.rcs(
                                to=message.from_,
                                from_="test",
                                text="Sorry, we had trouble enhancing the description. Using original description instead.",
                            )
                            description = description
                    else:
                        print(f"Astica API response: {result}")
                        client.send.rcs(
                            to=message.from_,
                            from_="test",
                            text="Sorry, we couldn't analyze the image. Please try again later.",
                        )

                except Exception as e:
                    print(f"Error with Astica API: {e}")
                    client.send.rcs(
                        to=message.from_,
                        from_="test",
                        text="Sorry, we had trouble analyzing the image. Please try again later.",
                    )

            except PIL.UnidentifiedImageError:
                client.send.rcs(
                    to=message.from_,
                    from_="test",
                    text="Sorry, we couldn't process this file. Please make sure you're sending a supported image format (JPG, PNG, HEIC, etc).",
                )

        except requests.RequestException as e:
            client.send.rcs(
                to=message.from_,
                from_="test",
                text="Sorry, I had trouble downloading the image. Please try again later.",
            )
            print(f"Error downloading image: {e}")

    def _handle_action_message(self, message: InboundActionMessage):
        print(f"Processing action message: {message.action_title}, {message.payload}")
        if message.payload == "WHAT_IS_PINNACLE":
            client.send.rcs(
                to=message.from_,
                from_="test",
                text="Pinnacle is the RCS API for developers.",
            )

    def _handle_location_message(self, message):
        print(f"Processing location message: {message.coordinates}")
        client.send.rcs(
            to=message.from_,
            from_="test",
            text="Your coordinates are: " + str(message.coordinates),
        )


@app.post("/")
async def webhook(request: Request) -> dict[str, str]:
    message_handler = MessageHandler(os.getenv("WEBHOOK_SIGNING_SECRET") or "")
    message_data = await request.json()
    message_handler.handle_message(
        message_data, request.headers.get("PINNACLE-SIGNING-SECRET") or ""
    )
    return {"message": "Message received"}


import uvicorn

uvicorn.run(app, host="0.0.0.0", port=8000)
