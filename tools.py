from crewai_tools import BaseTool
import os
import requests
import json
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import re
import threading

load_dotenv()

cache_lock = threading.Lock()

search_cache = {}

def search_multiple_page(
    search_query: str, num_results: int, num_pages: int = 1
) -> str:
    url = "https://google.serper.dev/search"
    headers = {
        "X-API-KEY": os.getenv('SERPER_API_KEY'),
        "Content-Type": "application/json",
    }

    all_results = []

    for page in range(1, num_pages + 1):
        cache_key = (search_query, page)
        with cache_lock:
            if cache_key in search_cache:
                results = search_cache[cache_key]
            else:
                payload = json.dumps({"q": search_query, "num": num_results, "page": page})
                search_results = requests.request("POST", url, headers=headers, data=payload)
                results = search_results.json()
                search_cache[cache_key] = results

        if "organic" in results and results["organic"]:
            all_results.extend(results["organic"])
        else:
            break

    return all_results

def scrape_website(website_url: str) -> str:
    website_content = fetch_website_contents(website_url)
    if website_content is None:
        return "Website is unable to reach."

    soup = BeautifulSoup(website_content, "html.parser")

    text_content = soup.get_text(separator=" ")

    copyright_patterns = [
        r'©\s*\d{2,4}.*?(?=All Rights Reserved|$)',
        r'©\s*(?:[\w\s&,.]+)?\d{2,4}\s*-\s*\d{2,4}\.? .*?(?=All Rights Reserved|$)',
        r'©\s*(?:[\w\s&,.]+)?\d{4}(?:\s*-\s*\d{4})?.*?(?=All Rights Reserved|$)',
        r'©\s*[\w\s,&.]+'
    ]

    for pattern in copyright_patterns:
        matches = re.findall(pattern, text_content, re.IGNORECASE)
        if matches:
            return matches[-1]
    return "Copyright information is not found."


class WebsiteScraper(BaseTool):
    name: str = "Website Scraper Tool"
    description: str = (
        "Scrapes the contents of a website based on the given URL. This tool can be used to extract the contents of the website."
    )

    def _run(self, website_url: str) -> str:
        website_content = scrape_website(website_url)
        return website_content

class SubsidiarySearchSerper(BaseTool):
    name: str = "Subsidiary Searcher Tool"
    description: str = (
        "Searches the web based on a search query for the latest results. Uses the Serper API. This also returns the contents of the search results."
    )

    def _run(self, search_query: str) -> str:
        search_results = search_multiple_page(search_query, 100)
        return search_results

class SearchSerper(BaseTool):
    name: str = "Company Searcher Tool"
    description: str = (
        "Searches the web based on a search query for the latest results. Uses the Serper API. This also returns the contents of the search results."
    )

    def _run(self, search_query: str) -> str:
        search_results = search_multiple_page(search_query, 10)
        return search_results
    
class CopyrightSearchSerper(BaseTool):
    name: str = "Copyright Searcher Tool"
    description: str = (
        "Searches the web based on a search query for the latest results. Uses the Serper API. This also returns the contents of the search results."
    )

    def _run(self, search_query: str) -> str:
        search_query = f'"{search_query}" -linkedin -quora -instagram -youtube -facebook -twitter -pinterest -snapchat -github -whatsapp -tiktok -reddit -news -blog -articles -forums -pdf -sec.gov -x.com -amazon -vimeo'
        search_results = search_multiple_page(search_query, 100)
        return search_results


class DomainSearchSerper(BaseTool):
    name: str = "Domain Searcher Tool"
    description: str = (
        "Searches the web based on a search query for the latest results. Uses the Serper API. This also returns the contents of the search results."
    )

    def _run(self, search_query: str) -> str:
        search_query = f"{search_query}"
        search_results = search_multiple_page(search_query, 100, 3)
        return search_results
