import os
import config
import pandas as pd
import json
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from langchain_openai import AzureChatOpenAI
import json_repair
import tldextract
from tools import search_multiple_page

load_dotenv()

default_llm = AzureChatOpenAI(
    azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
    azure_deployment=os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME'),
    openai_api_version=os.getenv('OPENAI_API_VERSION'),
    temperature=0
)

script_dir = os.path.dirname(os.path.abspath(__file__))

def extract_domain_name(url):
    extracted = tldextract.extract(url)
    domain = f"{extracted.domain}.{extracted.suffix}"
    return domain

def read_chunks_from_file(input_file):
  data_file_path = os.path.join(script_dir, "data", input_file)

  df = pd.read_excel(data_file_path, header=None)
  df = df.applymap(lambda x: str(x).replace(",", ""))
  data_string = ", ".join(df.values.flatten())
  entries_list = data_string.split(", ")
  return entries_list

def create_result_directory(output_folder):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    final_results_path = os.path.join(script_dir, "final_results")
    new_folder_path = os.path.join(final_results_path, output_folder)
    os.makedirs(new_folder_path, exist_ok=True)
    set_log_file(os.path.join(new_folder_path, "log.txt"))

def set_log_file(file_path):
    with open(file_path, 'w') as f:
        f.write('')

    config.log_file = file_path

def get_log_file():
    return config.log_file

def main():
    companies = [
        {
            'file': 'kemper_corporation.xlsx',
            'main_company': 'Kemper Corporation.',
            'output_folder': 'KemperCorporation'
        },
    ]

    expert_website_researcher_agent_1 = Agent(
        role="Expert Website Researcher",
        goal="Accurately identify the main website of the company {company_name} , which is a part of {main_company}.",
        verbose=True,
        llm=default_llm,
        allow_delegation=False,
        backstory="""
            You have been a part of {main_company} for many years and have a deep understanding of the company's operations and online presence.
            As a seasoned investigator in the digital realm, you are a skilled web researcher capable of finding accurate company websites using search engines and verifying the information.
            With years of being with {company_name}, you are well known about the ins and outs of this company.
            You know all the websites with copyright same as main website of these.
            You also are expert in google searching and using sites like crunchbase, and Pitch book, etc to find the company details and get the website.
            You are meticulus and organized, and you only provide correct and precise data i.e. websites that you have identified correctly.""",
    )

    sample_expert_website_researcher_output = {
        'subdisiary_name1': [
            'https://www.subdisiary_name1.com',
            'https://www.subdisiary_name1.org'
        ]
    }

    sample_expert_website_researcher_output = json.dumps(sample_expert_website_researcher_output)

    for company in companies:
        input_file = company['file']
        main_company = company['main_company']
        output_folder = company['output_folder']

        final_results = []

        file_company_list = read_chunks_from_file(input_file)
        create_result_directory(output_folder)

        for subsidiary in file_company_list:
            search_results1 = search_multiple_page(f"{subsidiary} a part of {main_company} official website", 10, 1)
            search_results2 = search_multiple_page(f"{subsidiary} official website", 10, 1)
            search_results3 = search_multiple_page(f"{subsidiary}", 10, 1)

            search_results = json.dumps(search_results1 + search_results2 + search_results3)

            expert_website_researcher_task_1 = Task(
                description=(
                    """
                        {search_results}

                        Your task is to identify all potential official websites for the company {company_name}, which is a subsidiary of {main_company}, based on the search results provided above.

                        Instructions:
                        1. Thoroughly review each search result to ensure accuracy.
                        2. Identify the most relevant and official websites associated with {company_name}.
                        3. Consider factors such as domain authority, content relevance, and official branding.
                        4. Ensure the websites are possible official websites of the given subsidiary.
                        6. Exclude unrelated third-party profiles (e.g., Bloomberg, Meta, LinkedIn, Pitchbook, App store) unless they are the primary online presence of the company.
                        7. List all identified websites in a clear and organized manner.

                        Sample Output:
                        {sample_expert_website_researcher_output}

                        Important notes:
                        - Every set of rules and steps mentioned above must be followed to get the required results.
                        - Do not provide any other texts or information in the output as it will not work with the further process.
                        - Do not include ``` or any other such characters in the output.
                    """
                ),
                agent=expert_website_researcher_agent_1,  # Assigning the task to the researcher,
                expected_output="All possible official website of the company. {company_name}",
            )

            expert_website_researcher_crew_1 = Crew(
                agents=[expert_website_researcher_agent_1],
                tasks=[expert_website_researcher_task_1],
                process=Process.sequential,
                verbose=1
            )

            results = expert_website_researcher_crew_1.kickoff(inputs={"company_name": subsidiary, "main_company": main_company, "search_results": search_results, "sample_expert_website_researcher_output": sample_expert_website_researcher_output})
            results = json_repair.loads(results.raw)
            
            final_results.append(results)
        
        print(final_results)

        data = []
        for result in final_results:
            for company, urls in result.items():
                for url in urls:
                    data.append({'Company Name': company, 'Website URL': url})

        df = pd.DataFrame(data, columns=['Company Name', 'Website URL'])

        df.to_excel('./final_results/' + output_folder + '/' + main_company + '.xlsx', engine='openpyxl', index=False)

        print("Data exported")

if __name__ == "__main__":
    main()