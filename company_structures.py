from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
import os
from scrapegraphai.graphs import SmartScraperGraph
from dotenv import load_dotenv
import json
from crewai import Agent, Crew, Task
import time
import json_repair
from tools import SubsidiarySearchSerper

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

class Agents:
    def __init__(self):
        self.subsidiary_search = SubsidiarySearchSerper()

    def subsidiaries_finder_linkresearcher_agent(self):
        return Agent(
            role="Link Researcher",
            goal="Gather links that can help identify the subsidiaries/trusts/entities/global operations/charitable organizations/companies with more than 50% partnership of the given company.",
            verbose=True,
            llm=azure_model,
            allow_delegation=False,
            tools=[self.subsidiary_search],
            backstory="""
    You are a skilled web researcher with expertise in finding relevant information online.
    Your task is to gather links that can help identify the web links that can help to find out the subsidiaries/trusts/entities/global operations/charitable organizations/companies with more than 50% partnership of the given company.
    You are proficient in using search engines like Google to find web pages related to the subsidiaries/trusts/entities/global operations/charitable organizations/companies with more than 50% partnership of companies.
    You know how to select multiple relevant URLs from search results that are likely to contain information about the subsidiaries/trusts/entities/global operations/charitable organizations/companies with more than 50% partnership of a company.
    """)

    def brands_finder_linkresearcher_agent(self):
        return Agent(
            role="Link Researcher",
            goal="Gather links that can help identify the brands/sub-brands/trusts/entities/global operations/charitable organizations/companies with more than 50% partnership of the given company.",
            verbose=True,
            llm=azure_model,
            allow_delegation=False,
            tools=[self.subsidiary_search],
            backstory="""
    You are a skilled web researcher with expertise in finding relevant information online.
    Your task is to gather links that can help identify the web links that can help to find out the brands/sub-brands/trusts/entities/global operations/charitable organizations/companies with more than 50% partnership of the given company.
    You are proficient in using search engines like Google to find web pages related to the brands/sub-brands/trusts/entities/global operations/charitable organizations/companies with more than 50% partnership of companies.
    You know how to select multiple relevant URLs from search results that are likely to contain information about the brands/sub-brands/trusts/entities/global operations/charitable organizations/companies with more than 50% partnership of a company.
    """)

    def acquisitions_finder_linkresearcher_agent(self):
        return Agent(
            role="Link Researcher",
            goal="Gather links that can help identify the acquisitions/trusts/entities/global operations/charitable organizations/companies with more than 50% partnership of the given company.",
            verbose=True,
            llm=azure_model,
            allow_delegation=False,
            tools=[self.subsidiary_search],
            backstory="""
    You are a skilled web researcher with expertise in finding relevant information online.
    Your task is to gather links that can help identify the web links that can help to find out the acquisitions/trusts/entities/global operations/charitable organizations/companies with more than 50% partnership of the given company.
    You are proficient in using search engines like Google to find web pages related to the acquisitions/trusts/entities/global operations/charitable organizations/companies with more than 50% partnership of companies.
    You know how to select multiple relevant URLs from search results that are likely to contain information about the acquisitions/trusts/entities/global operations/charitable organizations/companies with more than 50% partnership of a company.
    """)

class Tasks:
    def subsidiaries_finder_link_grabber_task(self, agent):
        return Task(
            description=(
                """
                    Your task is to gather links that can help identify the subsidiaries, trusts, entities, charitable organizations, and companies with more than 50% partnership of the given company.
                    Follow these steps to ensure accurate and relevant results:
                    - Use a search engine like Google to find web pages related to the subsidiaries, trusts, entities, global operations, charitable organizations, and companies with more than 50% partnership of the company.
                    - Use concise search queries (for example, "all subsidiaries of {company_name}").
                    - Select multiple relevant URLs from the search results that are likely to contain information about the subsidiaries, entities, trusts, charitable organizations, and companies with more than 50% partnership of {company_name}.
                    - Ensure the selected links are from reputable sources such as:
                        - Corporate Filings and Legal Documents (SEC, EDGAR)
                        - Company Websites
                        - Latest Annual reports of the company {company_name}
                        - Business Databases
                        - News and Press Releases about acquisitions/investments by the company {company_name}
                        - Social Media and LinkedIn of the company {company_name}
                        - Patents and Trademarks of the company {company_name}
                        - FOIA Requests (For Government Contractors)
                        - Networking and Industry Contacts
                        - Legal Databases
                        - Craft.co
                        - Zoominfo
                        - Pitchbook
                        - SEC.gov
                        - Bloomberg
                        - Reuters
                        - Hoovers
                        - MarketWatch
                        - Yahoo Finance
                        - Forbes
                        - Business Insider
                        - If no relevant links are found, return "[]".
                    - Make sure, those relevant urls should be unique and not repeated.

                    Output the results in the following JSON format:
                    ["URL1", "URL2", "URL3"]

                    Example:
                    ["https://www.example.com/subsidiaries-of-alphabet", "https://www.example.com/alphabet-brands"]

                    Important note: Every step mentioned above must be followed to get the required results.
                    Do not provide any other texts or information in the output as it will not work with the further process.
                    Do not include ``` or any other such characters in the output.
                """
            ),
            agent=agent,
            expected_output="A list of URLs that can help identify the subsidiaries/trusts/entities/global operations/charitable organizations/companies with more than 50% partnership of the given company.",
        )
    
    def brands_finder_link_grabber_task(self, agent):
        return Task(
            description=(
                """
                    Your task is to gather links that can help identify the brands, subbrands, trusts, entities, charitable organizations, and companies with more than 50% partnership of the given company.
                    Follow these steps to ensure accurate and relevant results:
                    - Use a search engine like Google to find web pages related to the brands, subbrands, trusts, entities, global operations, charitable organizations, and companies with more than 50% partnership of the company.
                    - Use concise search query (for example, "all brands of {company_name}").
                    - Select multiple relevant URLs from the search results that are likely to contain information about the brands, subbrands, entities, trusts, charitable organizations, and companies with more than 50% partnership of {company_name}.
                    - Ensure the selected links are from reputable sources such as:
                        - Corporate Filings and Legal Documents (SEC, EDGAR)
                        - Company Websites
                        - Latest Annual reports of the company {company_name}
                        - Business Databases
                        - News and Press Releases about acquisitions/investments by the company {company_name}
                        - Social Media and LinkedIn of the company {company_name}
                        - Patents and Trademarks of the company {company_name}
                        - FOIA Requests (For Government Contractors)
                        - Networking and Industry Contacts
                        - Legal Databases
                        - Craft.co
                        - Zoominfo
                        - Pitchbook
                        - SEC.gov
                        - Bloomberg
                        - Reuters
                        - Hoovers
                        - MarketWatch
                        - Yahoo Finance
                        - Forbes
                        - Business Insider
                        - If no relevant links are found, return "[]".
                    - Make sure, those relevant urls should be unique and not repeated.

                    Output the results in the following JSON format:
                    ["URL1", "URL2", "URL3"]

                    Example:
                    ["https://www.example.com/subsidiaries-of-alphabet", "https://www.example.com/alphabet-brands"]

                    Important note: Every step mentioned above must be followed to get the required results.
                    Do not provide any other texts or information in the output as it will not work with the further process.
                    Do not include ``` or any other such characters in the output.
                """
            ),
            agent=agent,
            expected_output="A list of URLs that can help identify the brands/sub-brands/trusts/entities/global operations/charitable organizations/companies with more than 50% partnership of the given company.",
        )

    def acquisitions_finder_link_grabber_task(self, agent):
        return Task(
            description=(
                """
                    Your task is to gather links that can help identify the acquisitions, trusts, entities, charitable organizations, and companies with more than 50% partnership of the given company.
                    Follow these steps to ensure accurate and relevant results:
                    - Use a search engine like Google to find web pages related to the acquisitions, trusts, entities, global operations, charitable organizations, and companies with more than 50% partnership of the company.
                    - Use concise search query (for example, "all acquisitions of {company_name}").
                    - Select multiple relevant URLs from the search results that are likely to contain information about the acquisitions, entities, trusts, charitable organizations, and companies with more than 50% partnership of {company_name}.
                    - Ensure the selected links are from reputable sources such as:
                        - Corporate Filings and Legal Documents (SEC, EDGAR)
                        - Company Websites
                        - Latest Annual reports of the company {company_name}
                        - Business Databases
                        - News and Press Releases about acquisitions/investments by the company {company_name}
                        - Social Media and LinkedIn of the company {company_name}
                        - Patents and Trademarks of the company {company_name}
                        - FOIA Requests (For Government Contractors)
                        - Networking and Industry Contacts
                        - Legal Databases
                        - Craft.co
                        - Zoominfo
                        - Pitchbook
                        - SEC.gov
                        - Bloomberg
                        - Reuters
                        - Hoovers
                        - MarketWatch
                        - Yahoo Finance
                        - Forbes
                        - Business Insider
                        - If no relevant links are found, return "[]".
                    - Make sure, those relevant urls should be unique and not repeated.

                    Output the results in the following JSON format:
                    ["URL1", "URL2", "URL3"]

                    Example:
                    ["https://www.example.com/subsidiaries-of-alphabet", "https://www.example.com/alphabet-brands"]

                    Important note: Every step mentioned above must be followed to get the required results.
                    Do not provide any other texts or information in the output as it will not work with the further process.
                    Do not include ``` or any other such characters in the output.
                """
            ),
            agent=agent,
            expected_output="A list of URLs that can help identify the acquisitions/trusts/entities/global operations/charitable organizations/companies with more than 50% partnership of the given company.",
        )


class SubsidiaryCrew:
    def __init__(
        self,
        main_company: str,
        agents: Agents = None,
        tasks: Tasks = None,
    ):
        self.main_company = main_company
        self.agents = agents
        self.tasks = tasks

    def research_links(self):
        try:
            max_retries = 3
            retry_wait = 6
            retries = 0

            while retries < max_retries:
                try:
                    subsidiary_finder_link_grabber_agent = self.agents.subsidiaries_finder_linkresearcher_agent()
                    subsidiary_finder_link_grabber_task = self.tasks.subsidiaries_finder_link_grabber_task(subsidiary_finder_link_grabber_agent)

                    brands_finder_link_grabber_agent = self.agents.brands_finder_linkresearcher_agent()
                    brands_finder_link_grabber_task = self.tasks.brands_finder_link_grabber_task(brands_finder_link_grabber_agent)

                    acquisitions_finder_link_grabber_agent = self.agents.acquisitions_finder_linkresearcher_agent()
                    acquisitions_finder_link_grabber_task = self.tasks.acquisitions_finder_link_grabber_task(acquisitions_finder_link_grabber_agent)

                    crew = Crew(
                        agents=[
                            subsidiary_finder_link_grabber_agent,
                            brands_finder_link_grabber_agent,
                            acquisitions_finder_link_grabber_agent
                        ],
                        tasks=[
                            subsidiary_finder_link_grabber_task,
                            brands_finder_link_grabber_task,
                            acquisitions_finder_link_grabber_task
                        ],
                        verbose=True,
                        cache=True
                    )
                    crew.kickoff(
                        inputs={"company_name": self.main_company}
                    )

                    subsidiary_finder_links = json_repair.loads(subsidiary_finder_link_grabber_task.output.raw_output)
                    brands_finder_links = json_repair.loads(brands_finder_link_grabber_task.output.raw_output)
                    acquisitions_finder_links = json_repair.loads(acquisitions_finder_link_grabber_task.output.raw_output)

                    return {
                        "subsidiary_finder_links": subsidiary_finder_links,
                        "brands_finder_links": brands_finder_links,
                        "acquisitions_finder_links": acquisitions_finder_links
                    }
                except Exception as e:
                    if "429" in str(e):
                        print(f"Rate limit exceeded, retrying in {retry_wait} seconds...")
                        time.sleep(retry_wait)
                        retry_wait *= 2  # Exponential backoff
                        retries += 1
                    else:
                        print(f"Error processing chunk: {e}")
                        return None

            if retries == max_retries:
                print("Max retries exceeded.")
                return None
            
        except Exception as e:
            print(f"Error processing chunk 2: {e}")
            return None

def get_company_structures(url, main_company, log_file_path):
    try :
        sample_json_output = json.dumps({'company_structure': ['Company 1', 'Company 2']})
        sample_json_output2 = json.dumps({'company_structure': None})

        prompt = f"""
            Work as a scraper tool to get a list of all subsidiaries, mergers, acquisitions, legal entities, brands, sub-brands, and other companies with more than 50% stake of company {main_company} from the given website.
            Carefully look for the names of these entities.
            A reward of $1 will be given for every correct entity identified and a penalty of $.75 will be applied for every incorrect entity given.
            Only give me the name of the company names in an array, do not include any other information.

            Sample Output: {sample_json_output}

            If no subsidiaries are found then return {sample_json_output2}
        """
        smart_search_graph = SmartScraperGraph(
            prompt=prompt,
            source = url,
            config=graph_config,
        )
        result = smart_search_graph.run()

        graph_exec_info = smart_search_graph.get_execution_info()

        return {
            'result': result,
            'exec_info': graph_exec_info
        }
    except Exception as e:
        with open(log_file_path['log'], 'a') as f:
            f.write(f"Exception when getting company structures using Scrapegraph AI: {e}")
        print(f"Exception when getting company structures using Scrapegraph AI: {e}")
        return {'company_structure': None}
    
def get_company_structures_for_private_company(main_company):
    agents = Agents()
    tasks = Tasks()

    trip_crew_subsidiary_research = SubsidiaryCrew(
      main_company=main_company,
      agents=agents,
      tasks=tasks,
    )

    print("Bikash")

    research_links_results = trip_crew_subsidiary_research.research_links()

    print(research_links_results)

    research_links_results = research_links_results["subsidiary_finder_links"] + research_links_results["brands_finder_links"] + research_links_results["acquisitions_finder_links"]

    research_links_results = list(set(research_links_results))

    print(research_links_results)

    # subsidiaries_name = {main_company}

    