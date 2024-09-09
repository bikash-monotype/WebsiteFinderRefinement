import os
import requests
import json
from dotenv import load_dotenv
from crewai_tools import BaseTool

load_dotenv()

def search_multiple_page(
    search_query: str, num_results: int, num_pages: int = 1, log_file_path = 'log.txt'
) -> str:
    credits = 0
    url = os.getenv('SERPER_API_URL')
    headers = {
        "X-API-KEY": os.getenv('SERPER_API_KEY'),
        "Content-Type": "application/json",
    }

    all_results = []

    for page in range(1, num_pages + 1):
        payload = json.dumps({"q": search_query, "num": num_results, "page": page})
        search_results = requests.request("POST", url, headers=headers, data=payload)
        results = search_results.json()

        if 'statusCode' in results:
            if results['statusCode'] != 200:
                with open(log_file_path, 'a') as f:
                    f.write(f"\n(Serper Error) Error processing search query {search_query}: {results['message']}")

        if "organic" in results and results["organic"]:
            all_results.extend(results["organic"])
            credits += results['credits']
        else:
            break

    return {
        'serper_credits': credits,
        'all_results': all_results
    }