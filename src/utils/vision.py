"""Google Cloud Vision labels, mapped onto the dataset vocabulary."""

import os
import re
from pathlib import Path

from dotenv import load_dotenv

from src.utils.base import Detector, image_b64, post_json

ALIASES = {
    "coffee_mug": ["coffee mug", "coffee cup", "mug"],
    "cup": ["cup"],
    "sunglasses": ["sunglasses", "glasses", "eyewear"],
    "camera": ["camera", "dslr", "slr"],
    "phone": ["smartphone", "cellphone", "mobile phone", "phone"],
    "laptop": ["laptop"],
    "blurred_laptop": ["laptop", "computer", "monitor", "screen"],
    "blurred_lights": ["bokeh", "lights", "light", "night"],
    "mountain": ["mountain", "snow", "peak"],
    "ocean": ["ocean", "sea", "beach"],
    "man": ["man", "boy", "male"],
    "woman": ["woman", "girl", "lady", "female"],
    "people": ["people", "crowd", "person"],
    "urban_architecture": ["architecture", "skyline", "cityscape", "building"],
    "bookshelf": ["bookshelf", "bookcase"],
    "flowers": ["flower"],
    "dessert": ["dessert", "cake", "pastry"],
    "water_bottle": ["water bottle"],
    "perfume_bottle": ["perfume", "fragrance"],
    "flower_vase": ["vase"],
    "alarm_clock": ["alarm clock", "clock"],
    "handbag": ["handbag", "purse"],
    "kettle": ["kettle", "teapot"],
    "pen": ["fountain pen", "pen"],
    "lamp": ["lamp"],
    "book": ["book"],
    "soap": ["soap"],
    "jar": ["jar"],
    "towel": ["towel"],
    "horse": ["horse"],
    "Barack_Obama": ["barack obama", "obama"],
    "Donald_Trump": ["donald trump", "trump"],
    "Joe_Biden": ["joe biden", "biden"],
    "Ben_Affleck": ["ben affleck", "affleck"],
    "Brad_Pitt": ["brad pitt"],
    "Jeff_Bezos": ["jeff bezos", "bezos"],
    "White_House": ["white house"],
    "Oval_Office": ["oval office"],
    "United_Nations": ["united nations"],
    "Venice_Film_Festival": ["venice film"],
    "White_House_Correspondents_Dinner": ["correspondents dinner"],
    "State_of_the_Union": ["state of the union"],
    "Panama_City": ["panama"],
    "Antalya": ["antalya"],
    "Florida": ["florida"],
    "Buffalo": ["buffalo"],
}


def _has(text: str, key: str) -> bool:
    return re.search(rf"(?<![a-z]){re.escape(key)}s?(?![a-z])", text) is not None


def _pick(names: list[str], choices: list[str]) -> str:
    for name in names:
        low = name.lower()
        found, best = "", 0
        for choice in choices:
            keys = ALIASES.get(choice, [choice.lower().replace("_", " ")])
            for key in keys:
                if _has(low, key) and len(key) > best:
                    found, best = choice, len(key)
        if found:
            return found
    return ""


def _area(obj: dict) -> float:
    vertices = obj["boundingPoly"]["normalizedVertices"]
    xs = [vertex.get("x", 0) for vertex in vertices]
    ys = [vertex.get("y", 0) for vertex in vertices]
    return (max(xs) - min(xs)) * (max(ys) - min(ys))


def _descriptions(ann: dict) -> tuple[list[str], list[str]]:
    objects = [obj["name"] for obj in sorted(ann.get("localizedObjectAnnotations", []), key=_area)]
    labels = [item["description"] for item in ann.get("labelAnnotations", [])]
    web = ann.get("webDetection", {})
    labels += [item["description"] for item in web.get("webEntities", []) if item.get("description")]
    labels += [item["label"] for item in web.get("bestGuessLabels", [])]
    labels += [item.get("pageTitle", "") for item in web.get("pagesWithMatchingImages", [])]
    labels += [item["description"] for item in ann.get("landmarkAnnotations", [])]
    return objects, labels


class CloudVision(Detector):
    name = "cloud-vision_google"

    def __init__(self):
        load_dotenv()
        self.key = os.environ["GOOGLE_API_KEY"]

    def predict(self, path: Path, vocab: dict[str, list[str]]) -> dict[str, str]:
        body = {"requests": [{"image": {"content": image_b64(path)}, "features": [
            {"type": "OBJECT_LOCALIZATION", "maxResults": 8},
            {"type": "LABEL_DETECTION", "maxResults": 15},
            {"type": "WEB_DETECTION", "maxResults": 15},
            {"type": "LANDMARK_DETECTION", "maxResults": 5},
        ]}]}
        ann = post_json(
            "https://vision.googleapis.com/v1/images:annotate?key=" + self.key,
            body,
        )["responses"][0]
        objects, labels = _descriptions(ann)
        foreground = _pick(objects, vocab["foreground"]) or _pick(labels, vocab["foreground"])
        background = _pick([name for name in labels if name], [choice for choice in vocab["background"] if choice != foreground])
        relation = _relation(objects, labels, vocab["relation"])
        return {"foreground": foreground, "background": background, "relation": relation}


def _relation(objects: list[str], labels: list[str], choices: list[str]) -> str:
    relation = _pick(objects + labels, choices)
    if relation:
        return relation
    if len(choices) == 1:
        return choices[0]
    if not _pick(objects + labels, ["people"]):
        return ""
    blob = " ".join(objects + labels).lower()
    if re.search(r"event|festival|dinner|ceremony|party|speech|convention", blob):
        return "people_at_event"
    return "people_at_location"
