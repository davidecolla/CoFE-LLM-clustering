# © Copyright European Union - 2026

import os
from dotenv import load_dotenv

# Load environment variables before importing other modules
load_dotenv()

from src.graph import GraphAgent
from langchain_core.runnables.config import RunnableConfig
from langchain_core.messages import HumanMessage
import json

# Resolve data paths relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")


if __name__ == "__main__":
    graph = GraphAgent().build_graph()
    img = graph.get_graph().draw_mermaid_png()
    with open(os.path.join(BASE_DIR, "graph.png"), 'wb') as file:
        file.write(img)
    exit()
    print("=== Running Agent ===")

    question = f"""I have 8000 texts coming from European Consultations. 
    I would like to cluster them according to the topics they belong. 
    There should be 10 clusters.
    The text might be expressed in one of the 27 languages of European Union member states.
    The language of each document is know and validated. Do not validate again the language.
    Each entry of the dataset has a JSON schema. You will find the language for each document encoded in an input field.
    The dataset is stored in a json file you can find at {os.path.join(DATA_DIR, "proposals_tiny.json")}
    The json schema of each entry is as follows:
    {{
        "text": "The text of the document",
        "id": "id of the document",
        "language": "The two-letters language code of the document eg. en, fr, de, etc..."
    }}
    If you need to translate the documents in English you can find the translation already computed here
    {os.path.join(DATA_DIR, "proposals_tiny_translation.json")}. For each entry, if the 
    language is not 'en' the field 'text_en' has been added. Such field includes the English translation
    of the original text. 
    I want as output a json file including a list of object, one for each input text.
    Each json object must have the following structure:
    {{
        "id": "id of the document",
        "cluster": "name of the cluster"
    }}
    Please provide me also an evaluation of clusters using silhouette in a dedicated file, called results.txt.
    I want to have both the silhouette for each sample, but also I want to have the average silhouette.
    Output files should be saved in the current directory.
    """

    config = RunnableConfig(
        {
            "configurable": {
                "thread_id": "1",
            },
        }
    )

    for event in graph.stream(
        {"question": [HumanMessage(question)], "messages": []},
        config=config
    ):
        for value in event.values():
            print("\n\n === Assistant ===\n", value)
            if isinstance(value, dict) and value.get("plan", False):
                json.dump(
                    value['plan'],
                    open(os.path.join(DATA_DIR, "plan.json"), "w"),
                    indent=2, ensure_ascii=False
                )
