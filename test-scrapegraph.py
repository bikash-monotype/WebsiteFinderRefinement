from scrapegraphai.graphs import SmartScraperGraph
from langchain_openai import ChatOpenAI
import json
import os

os.environ['OPENAI_API_KEY'] = 'sk-1234567890abcdef1234567890abcdef'

default_llm = ChatOpenAI(
    model='llama3.1:8b',
    base_url='http://localhost:11434/v1/',
)

graph_config = {
    "llm": {
        "model_instance": default_llm,
        "model_tokens": 100000,
        "base_url": "http://localhost:11434",
    },
    "embeddings": {
        "model": "nomic-embed-text",
        "temperature": 0,
        "base_url": "http://localhost:11434",
    },
    "verbose": True,
    "headless": False
}

sample_json_output = json.dumps({'company_structure': ['Company 1', 'Company 2']})
sample_json_output2 = json.dumps({'company_structure': None})

prompt = f"""
    Work as a scraper tool to get a list of all subsidiaries, mergers, acquisitions, legal entities, brands, sub-brands, and other companies with more than 50% stake of company "User Testing Inc." from the given website.
    Carefully look for the names of these entities.
    A reward of $1 will be given for every correct entity identified and a penalty of $.75 will be applied for every incorrect entity given.
    Only give me the name of the company names in an array, do not include any other information.

    Sample Output: {sample_json_output}

    If no subsidiaries are found then return {sample_json_output2}
"""
smart_search_graph = SmartScraperGraph(
    prompt=(
        "Retrieve all subsidiaries of User Testing Inc. as listed in their latest SEC filing. "
        "Return the results exclusively in a JSON array format containing the names of these subsidiaries. "
        "Ensure strict compliance with the JSON array format to avoid penalties."
        "If no subsidiaries are found, return an empty array."
    ),
    source="https://www.sec.gov/Archives/edgar/data/1557127/000162828021020026/usertestingincs-1.htm",
    config=graph_config,
)

try:
    result = smart_search_graph.run()
    print(result)
except json.JSONDecodeError as e:
    print(f"JSONDecodeError: {e}")
    print("The output from the LLM was not valid JSON.")
except Exception as e:
    print(f"An error occurred: {e}")