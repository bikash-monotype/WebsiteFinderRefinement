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
            From the footer area of the provided webpage, extract the copyright text. 
            Ensure the following:
            1. Do not include 'All rights reserved' in the output.
            2. If the copyright text is not found in the webpage, return {'copyright': None}.
            3. The output should be in the format: {'copyright': 'Copyright © YEAR Company.'}
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