from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
import os
from scrapegraphai.graphs import SmartScraperGraph
from dotenv import load_dotenv

load_dotenv()

azure_model = AzureChatOpenAI(
    openai_api_version=os.getenv('OPENAI_API_VERSION'),
    azure_deployment=os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME'),
    azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
    api_key=os.getenv('AZURE_OPENAI_API_KEY'),
    temperature=0
)

azure_embeddings = AzureOpenAIEmbeddings(
    azure_deployment=os.getenv('AZURE_OPENAI_EMBEDDINGS'),
    openai_api_version="2023-05-15",
    azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
    api_key=os.getenv('AZURE_OPENAI_API_KEY'),
)

os.environ['AZURE_OPENAI_ENDPOINT'] = os.getenv('AZURE_OPENAI_ENDPOINT')
os.environ['AZURE_OPENAI_CHAT_DEPLOYMENT_NAME'] = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')
os.environ['MODEL_NAME'] = 'Azure'
os.environ['AZURE_OPENAI_API_KEY'] = os.getenv('AZURE_OPENAI_API_KEY')
os.environ['AZURE_OPENAI_API_VERSION'] = '2023-08-01-preview'
os.environ['AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME'] = os.getenv('AZURE_OPENAI_EMBEDDINGS')

graph_config = {
    "llm": {
        "model_instance": azure_model,
        "model_tokens": 100000,
    },
    "embeddings": {
        "model_instance": azure_embeddings
    },
    "verbose": True,
    "headless": False,
}

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
            f.write(f"Exception when getting copyright using scrapegraph AI: {e}")
        print(f"Exception when getting copyright using scrapegraph AI: {e}")
        return {'copyright': None}