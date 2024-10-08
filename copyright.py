from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
import os
from scrapegraphai.graphs import SmartScraperGraph
from dotenv import load_dotenv
from helpers import get_scrapegraph_config

load_dotenv()

os.environ['AZURE_OPENAI_ENDPOINT'] = os.getenv('AZURE_OPENAI_ENDPOINT')
os.environ['AZURE_OPENAI_CHAT_DEPLOYMENT_NAME'] = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')
os.environ['MODEL_NAME'] = 'Azure'
os.environ['AZURE_OPENAI_API_KEY'] = os.getenv('AZURE_OPENAI_API_KEY')
os.environ['AZURE_OPENAI_API_VERSION'] = '2023-08-01-preview'
os.environ['AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME'] = os.getenv('AZURE_OPENAI_EMBEDDINGS')

graph_config = get_scrapegraph_config()

def get_copyright(url, log_file_path):
    try :
        prompt = """
            From the footer area of the provided webpage, extract the copyright text if it exists.

            Ensure the following:
            1. Do not include 'All rights reserved' in the output.
            2. If no copyright text is found on the webpage, return {'copyright': None}.
            3. Only extract the copyright text if it clearly indicates ownership, in the format similar to:
            - "© YEAR Company Name."
            - If variations such as "issued by" or "licensed to" are found instead, treat them as **not valid** for ownership, and return {'copyright': None}.
            4. The output format must be: {'copyright': 'Copyright © YEAR Company.'} if the correct copyright is found.

            Important: Do not assume the presence of copyright text. Ensure it actually exists on the webpage.
        """
        smart_scraper_graph = SmartScraperGraph(
            prompt=prompt,
            source = url,
            config=graph_config,
        )

        result = smart_scraper_graph.run()
        graph_exec_info = smart_scraper_graph.get_execution_info()

        return {
            'result': result,
            'exec_info': graph_exec_info
        }
    except Exception as e:
        with open(log_file_path['log'], 'a') as f:
            f.write(f"Exception when getting copyright from {url} using scrapegraph AI: {e}")
        print(f"Exception when getting copyright from {url} using scrapegraph AI: {e}")
        return {
            'result': { 'copyright': None },
            'exec_info': None
        }