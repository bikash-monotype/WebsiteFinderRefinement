from langchain_openai import AzureChatOpenAI
import os
from scrapegraphai.graphs import SmartScraperGraph
from dotenv import load_dotenv
import json
from crewai import Agent, Task, Crew
from tools import search_multiple_page
from helpers import remove_trailing_slash, get_scrapegraph_config, tokenize_text
import json_repair
import time

load_dotenv()

default_llm = AzureChatOpenAI(
    azure_endpoint='',
    model=os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME'),
    openai_api_version=os.getenv('OPENAI_API_VERSION'),
    temperature=0
)

os.environ['AZURE_OPENAI_ENDPOINT'] = os.getenv('AZURE_OPENAI_ENDPOINT')
os.environ['AZURE_OPENAI_CHAT_DEPLOYMENT_NAME'] = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')
os.environ['MODEL_NAME'] = 'Azure'
os.environ['AZURE_OPENAI_API_KEY'] = os.getenv('AZURE_OPENAI_API_KEY')
os.environ['AZURE_OPENAI_API_VERSION'] = '2023-08-01-preview'
os.environ['AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME'] = os.getenv('AZURE_OPENAI_EMBEDDINGS')

graph_config = get_scrapegraph_config()

subsidiary_finder_link_grabber_agent = Agent(
    role="Link Researcher",
    goal="Gather links that can help identify the subsidiaries/trusts/entities/global operations/charitable organizations/companies with more than 50% partnership of the given company.",
    verbose=True,
    llm=default_llm,
    allow_delegation=False,
    backstory="""
    You are a skilled web researcher with expertise in finding relevant information online.
    Your task is to gather links that can help identify the web links that can help to find out the subsidiaries/trusts/entities/global operations/charitable organizations/companies with more than 50% partnership of the given company.
    You are proficient in using search engines like Google to find web pages related to the subsidiaries/trusts/entities/global operations/charitable organizations/companies with more than 50% partnership of companies.
    You know how to select multiple relevant URLs from search results that are likely to contain information about the subsidiaries/trusts/entities/global operations/charitable organizations/companies with more than 50% partnership of a company.
""")

brands_finder_link_grabber_agent = Agent(
    role="Link Researcher",
    goal="Gather links that can help identify the brands/sub-brands/trusts/entities/global operations/charitable organizations/companies with more than 50% partnership of the given company.",
    verbose=True,
    llm=default_llm,
    allow_delegation=False,
    backstory="""
    You are a skilled web researcher with expertise in finding relevant information online.
    Your task is to gather links that can help identify the web links that can help to find out the brands/sub-brands/trusts/entities/global operations/charitable organizations/companies with more than 50% partnership of the given company.
    You are proficient in using search engines like Google to find web pages related to the brands/sub-brands/trusts/entities/global operations/charitable organizations/companies with more than 50% partnership of companies.
    You know how to select multiple relevant URLs from search results that are likely to contain information about the brands/sub-brands/trusts/entities/global operations/charitable organizations/companies with more than 50% partnership of a company.
""")

acquisitions_finder_link_grabber_agent = Agent(
    role="Link Researcher",
    goal="Gather links that can help identify the acquisitions/trusts/entities/global operations/charitable organizations/companies with more than 50% partnership of the given company.",
    verbose=True,
    llm=default_llm,
    allow_delegation=False,
    backstory="""
    You are a skilled web researcher with expertise in finding relevant information online.
    Your task is to gather links that can help identify the web links that can help to find out the acquisitions/trusts/entities/global operations/charitable organizations/companies with more than 50% partnership of the given company.
    You are proficient in using search engines like Google to find web pages related to the acquisitions/trusts/entities/global operations/charitable organizations/companies with more than 50% partnership of companies.
    You know how to select multiple relevant URLs from search results that are likely to contain information about the acquisitions/trusts/entities/global operations/charitable organizations/companies with more than 50% partnership of a company.
    """)
        
def get_links_for_company_structures(main_company, log_file_path, url):
    try :
        sample_json_output = json.dumps({'links': ['https://example1.com', 'https://example2.com']})
        sample_json_output2 = json.dumps({'links': None})

        prompt = f"""
            Act as a scraper tool to extract all the links from the given website that might contain information about subsidiaries, mergers, acquisitions, legal entities, brands, sub-brands, or other companies where {main_company} holds more than 50% ownership. Specifically target links that could lead to sections related to these entities or relevant financial or legal information.
            Look for the text such as "Subsidiaries", "Mergers", "Acquisitions", "Legal Entities", "Brands", "Sub Brands" in the provided website.
            Provide the links in an array format, without any additional details.
            A reward of $1 will be given for each correctly identified entity, and a penalty of $0.75 will be applied for each incorrect entity.
            Sample Output: {sample_json_output}
            If no relevant links are found, return: {sample_json_output2}
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
            f.write(f"Exception when getting links for company structures using Scrapegraph AI for {url}: {e}")
        print(f"Exception when getting links for company structures using Scrapegraph AI for {url}: {e}")
        return {'result': {'company_structure': None}, 'exec_info': None}

def get_links_for_company_structures_for_private_company(main_company, log_file_path):
    try:
        subsidiary_finder_search_results = search_multiple_page("all subsidiaries of " + main_company, 100, 3, log_file_path)
        brands_finder_search_results = search_multiple_page("all brands of " + main_company, 100, 3, log_file_path)
        acquisitions_finder_search_results = search_multiple_page("all acquisitions of " + main_company, 100, 3, log_file_path)

        subsidiary_finder_link_grabber_task = Task(
            description=(
                f"""
                    ```{subsidiary_finder_search_results['all_results']}```

                    From the above provided context, your task is to gather links that can help identify the subsidiaries, trusts, entities, charitable organizations, and companies with more than 50% partnership of the given company.
                    Follow these steps to ensure accurate and relevant results:
                    - Select multiple relevant URLs from the provided context that are likely to contain information about the subsidiaries, entities, trusts, charitable organizations, and companies with more than 50% partnership of {main_company}.
                    - Ensure the selected links are from reputable sources such as:
                        - Corporate Filings and Legal Documents (SEC, EDGAR)
                        - Company Websites
                        - Latest Annual reports of the company {main_company}
                        - Business Databases
                        - News and Press Releases about acquisitions/investments by the company {main_company}
                        - Social Media and LinkedIn of the company {main_company}
                        - Patents and Trademarks of the company {main_company}
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
            agent=subsidiary_finder_link_grabber_agent,
            expected_output="A list of URLs that can help identify the subsidiaries/trusts/entities/global operations/charitable organizations/companies with more than 50% partnership of the given company.",
        )

        brands_finder_link_grabber_task = Task(
            description=(
                f"""
                    ```{brands_finder_search_results['all_results']}```

                    From the above provided context, your task is to gather links that can help identify the brands, subbrands, trusts, entities, charitable organizations, and companies with more than 50% partnership of the given company.
                    Follow these steps to ensure accurate and relevant results:
                    - Select multiple relevant URLs from the provided context that are likely to contain information about the brands, subbrands, entities, trusts, charitable organizations, and companies with more than 50% partnership of {main_company}.
                    - Ensure the selected links are from reputable sources such as:
                        - Corporate Filings and Legal Documents (SEC, EDGAR)
                        - Company Websites
                        - Latest Annual reports of the company {main_company}
                        - Business Databases
                        - News and Press Releases about acquisitions/investments by the company {main_company}
                        - Social Media and LinkedIn of the company {main_company}
                        - Patents and Trademarks of the company {main_company}
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
            agent=brands_finder_link_grabber_agent,
            expected_output="A list of URLs that can help identify the brands/sub-brands/trusts/entities/global operations/charitable organizations/companies with more than 50% partnership of the given company.",
        )

        acquisitions_finder_link_grabber_task = Task(
                description=(
                    f"""
                        ```{acquisitions_finder_search_results['all_results']}```

                        From the above provided context, your task is to gather links that can help identify the acquisitions, trusts, entities, charitable organizations, and companies with more than 50% partnership of the given company.
                        Follow these steps to ensure accurate and relevant results:
                        - Select multiple relevant URLs from the provided context that are likely to contain information about the acquisitions, entities, trusts, charitable organizations, and companies with more than 50% partnership of {main_company}.
                        - Ensure the selected links are from reputable sources such as:
                            - Corporate Filings and Legal Documents (SEC, EDGAR)
                            - Company Websites
                            - Latest Annual reports of the company {main_company}
                            - Business Databases
                            - News and Press Releases about acquisitions/investments by the company {main_company}
                            - Social Media and LinkedIn of the company {main_company}
                            - Patents and Trademarks of the company {main_company}
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
                agent=acquisitions_finder_link_grabber_agent,
                expected_output="A list of URLs that can help identify the acquisitions/trusts/entities/global operations/charitable organizations/companies with more than 50% partnership of the given company.",
            )

        subsidiary_finder_prompt_tokens = tokenize_text(
            f"""
                Link Researcher
                Gather links that can help identify the brands/sub-brands/trusts/entities/global operations/charitable organizations/companies with more than 50% partnership of the given company.
                You are a skilled web researcher with expertise in finding relevant information online.
                Your task is to gather links that can help identify the web links that can help to find out the subsidiaries/trusts/entities/global operations/charitable organizations/companies with more than 50% partnership of the given company.
                You are proficient in using search engines like Google to find web pages related to the subsidiaries/trusts/entities/global operations/charitable organizations/companies with more than 50% partnership of companies.
                You know how to select multiple relevant URLs from search results that are likely to contain information about the subsidiaries/trusts/entities/global operations/charitable organizations/companies with more than 50% partnership of a company.

                ```{subsidiary_finder_search_results['all_results']}```

                From the above provided context, your task is to gather links that can help identify the subsidiaries, trusts, entities, charitable organizations, and companies with more than 50% partnership of the given company.
                Follow these steps to ensure accurate and relevant results:
                - Select multiple relevant URLs from the provided context that are likely to contain information about the subsidiaries, entities, trusts, charitable organizations, and companies with more than 50% partnership of {main_company}.
                - Ensure the selected links are from reputable sources such as:
                    - Corporate Filings and Legal Documents (SEC, EDGAR)
                    - Company Websites
                    - Latest Annual reports of the company {main_company}
                    - Business Databases
                    - News and Press Releases about acquisitions/investments by the company {main_company}
                    - Social Media and LinkedIn of the company {main_company}
                    - Patents and Trademarks of the company {main_company}
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
        )

        brands_finder_prompt_tokens = tokenize_text(
            f"""
                Link Researcher
                Gather links that can help identify the brands/sub-brands/trusts/entities/global operations/charitable organizations/companies with more than 50% partnership of the given company.
                You are a skilled web researcher with expertise in finding relevant information online.
                Your task is to gather links that can help identify the web links that can help to find out the brands/sub-brands/trusts/entities/global operations/charitable organizations/companies with more than 50% partnership of the given company.
                You are proficient in using search engines like Google to find web pages related to the brands/sub-brands/trusts/entities/global operations/charitable organizations/companies with more than 50% partnership of companies.
                You know how to select multiple relevant URLs from search results that are likely to contain information about the brands/sub-brands/trusts/entities/global operations/charitable organizations/companies with more than 50% partnership of a company.

                ```{brands_finder_search_results['all_results']}```

                From the above provided context, your task is to gather links that can help identify the brands, subbrands, trusts, entities, charitable organizations, and companies with more than 50% partnership of the given company.
                Follow these steps to ensure accurate and relevant results:
                - Select multiple relevant URLs from the provided context that are likely to contain information about the brands, subbrands, entities, trusts, charitable organizations, and companies with more than 50% partnership of {main_company}.
                - Ensure the selected links are from reputable sources such as:
                    - Corporate Filings and Legal Documents (SEC, EDGAR)
                    - Company Websites
                    - Latest Annual reports of the company {main_company}
                    - Business Databases
                    - News and Press Releases about acquisitions/investments by the company {main_company}
                    - Social Media and LinkedIn of the company {main_company}
                    - Patents and Trademarks of the company {main_company}
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
        )

        acquisitions_finder_prompt_tokens = tokenize_text(
            f"""
                Link Researcher
                Gather links that can help identify the acquisitions/trusts/entities/global operations/charitable organizations/companies with more than 50% partnership of the given company.
                You are a skilled web researcher with expertise in finding relevant information online.
                Your task is to gather links that can help identify the web links that can help to find out the acquisitions/trusts/entities/global operations/charitable organizations/companies with more than 50% partnership of the given company.
                You are proficient in using search engines like Google to find web pages related to the acquisitions/trusts/entities/global operations/charitable organizations/companies with more than 50% partnership of companies.
                You know how to select multiple relevant URLs from search results that are likely to contain information about the acquisitions/trusts/entities/global operations/charitable organizations/companies with more than 50% partnership of a company.

                ```{acquisitions_finder_search_results['all_results']}```

                From the above provided context, your task is to gather links that can help identify the acquisitions, trusts, entities, charitable organizations, and companies with more than 50% partnership of the given company.
                Follow these steps to ensure accurate and relevant results:
                - Select multiple relevant URLs from the provided context that are likely to contain information about the acquisitions, entities, trusts, charitable organizations, and companies with more than 50% partnership of {main_company}.
                - Ensure the selected links are from reputable sources such as:
                    - Corporate Filings and Legal Documents (SEC, EDGAR)
                    - Company Websites
                    - Latest Annual reports of the company {main_company}
                    - Business Databases
                    - News and Press Releases about acquisitions/investments by the company {main_company}
                    - Social Media and LinkedIn of the company {main_company}
                    - Patents and Trademarks of the company {main_company}
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
        )

        trip_crew_subsidiary_research = Crew(
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
            allow_delegation=False,
            cache=True
        )
        time.sleep(3)
        trip_crew_subsidiary_research.kickoff()

        subsidiary_finder_completions_tokens = tokenize_text(
            f"""
                Thought: I now can give a great answer
                Final Answer: {str(subsidiary_finder_link_grabber_task.output.raw)}
            """
        )

        brands_finder_completions_tokens = tokenize_text(
            f"""
                Thought: I now can give a great answer
                Final Answer: {str(brands_finder_link_grabber_task.output.raw)}
            """
        )

        acquisitions_finder_completions_tokens = tokenize_text(
            f"""
                Thought: I now can give a great answer
                Final Answer: {str(acquisitions_finder_link_grabber_task.output.raw)}
            """
        )

        llm_usage = {
            'prompt_tokens': subsidiary_finder_prompt_tokens + brands_finder_prompt_tokens + acquisitions_finder_prompt_tokens,
            'completion_tokens': subsidiary_finder_completions_tokens + brands_finder_completions_tokens + acquisitions_finder_completions_tokens
        }

        subsidiary_finder_links = json_repair.loads(subsidiary_finder_link_grabber_task.output.raw)
        brands_finder_links = json_repair.loads(brands_finder_link_grabber_task.output.raw)
        acquisitions_finder_links = json_repair.loads(acquisitions_finder_link_grabber_task.output.raw)

        subsidiary_finder_links = remove_trailing_slash(subsidiary_finder_links)
        brands_finder_links = remove_trailing_slash(brands_finder_links)
        acquisitions_finder_links = remove_trailing_slash(acquisitions_finder_links)

        research_links_results = set(subsidiary_finder_links) | set(brands_finder_links) | set(acquisitions_finder_links)

        return {
            'links': list(research_links_results),
            'serper_credits': subsidiary_finder_search_results['serper_credits'] + brands_finder_search_results['serper_credits'] + acquisitions_finder_search_results['serper_credits'],
            'llm_usage': {
                'prompt_tokens': llm_usage['prompt_tokens'],
                'completion_tokens': llm_usage['completion_tokens']
            },
        }
    except Exception as e:
        with open(log_file_path, 'a') as f:
            f.write(f"Exception when getting links for company structures for private company: {e}")
        print(f"Exception when getting links for company structures for private company: {e}")
        return {'links': None, 'serper_credits': 0, 'llm_usage':{ 
            'prompt_tokens': 0,
            'completion_tokens': 0
         }}

def get_company_structures(main_company, log_file_path, url):
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
            source=url,
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
            f.write(f"Exception when getting company structures using Scrapegraph AI for {url}: {e}")
        print(f"Exception when getting company structures using Scrapegraph AI for {url}: {e}")
        return {'result': {'company_structure': None}, 'exec_info': None}

    