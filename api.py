from typing import Optional, List
from fastapi import FastAPI
from pydantic import BaseModel

from get_tags import predict

app = FastAPI()

class Text(BaseModel):
    title: str
    body: str


class Texts(BaseModel):
    title: List[str]
    body: List[str]


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/get_tags_single")
def read_item(text: Text):
    """ returns tag predictions for single pair of text and body """

    predictions = predict([[text.title], [text.body]])

    return predictions


@app.post("/get_tags_mutliple")
def read_item(texts: Texts):
    """ returns tag predictions for multiple pairs of text and body """

    predictions = predict([texts.title, texts.body])

    return predictions
