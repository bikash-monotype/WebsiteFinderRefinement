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
import urllib.parse
import time
from langchain_openai import AzureChatOpenAI
from copyright import get_copyright
import multiprocessing
from helpers import set_log_file, get_log_file, get_links, extract_year
import datetime
from openpyxl import load_workbook
import re

load_dotenv()

default_llm = AzureChatOpenAI(
    azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
    model=os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME'),
    openai_api_version=os.getenv('OPENAI_API_VERSION'),
    temperature=0
)

script_dir = os.path.dirname(os.path.abspath(__file__))

def extract_domain_name(url):
    parsed_url = urllib.parse.urlparse(url)
    extracted = tldextract.extract(parsed_url.netloc)
    domain_name = f"{extracted.domain}.{extracted.suffix}"

    return domain_name

def process_website(website):
    domain_name = extract_domain_name(website)
    copyright_info = get_copyright(website)
    return (domain_name, copyright_info)

def extract_main_part(url):
    parsed_url = urllib.parse.urlparse(url)
    extracted = tldextract.extract(parsed_url.netloc)
    domain_name = extracted.domain

    return domain_name

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

def process_copyright_research(row):
    company_name = row['Company Name']
    copyright = row['Copyright']
    copyright_results = set()

    year = extract_year(copyright)

    copyright_result1 = search_multiple_page(
        f'"{copyright}" -linkedin -quora -instagram -youtube -facebook -twitter -pinterest -snapchat -github -whatsapp -tiktok -reddit -x.com -amazon -vimeo', 100, 3)
    
    if year is not None:
        copyright_result2 = search_multiple_page(
            f'"© {year} {company_name}" -linkedin -quora -instagram -youtube -facebook -twitter -pinterest -snapchat -github -whatsapp -tiktok -reddit -x.com -amazon -vimeo', 100, 3)
    else:
        copyright_result2 = []

    words = copyright.split()
    filtered_words = [word for word in words if not re.search(r'\b(group|ltd)\b', word, re.IGNORECASE)]
    filtered_copyright = ' '.join(filtered_words)
    filtered_copyright_ran_already = False

    if filtered_copyright != copyright:
        filtered_copyright_ran_already = True
        copyright_result3 = search_multiple_page(
            f'"{filtered_copyright}" -linkedin -quora -instagram -youtube -facebook -twitter -pinterest -snapchat -github -whatsapp -tiktok -reddit -x.com -amazon -vimeo', 100, 3)
    else:
        copyright_result3 = []

    copyright_result4 = []

    if ( year is not None and filtered_copyright != ("© "+ year + " " + company_name)):
        if filtered_copyright_ran_already is False:
            copyright_result4 = search_multiple_page(
                f'"{filtered_copyright}" -linkedin -quora -instagram -youtube -facebook -twitter -pinterest -snapchat -github -whatsapp -tiktok -reddit -x.com -amazon -vimeo', 100, 3)           
    
    copyright_result = copyright_result1 + copyright_result2 + copyright_result3 + copyright_result4

    for result in copyright_result:
        try:
            copyright_results.add(extract_domain_name(result['link']))

            if "sitelinks" in result:
                for sitelink in result["sitelinks"]:
                    copyright_results.add(extract_domain_name(sitelink['link']))
        except KeyError:
            continue

    return copyright_results

def process_domain_research(main_part):
    domain_search_results = set()
    search_results1 = search_multiple_page(f"site:{main_part}.*", 100, 3)
    search_results2 = search_multiple_page(f"site:{main_part}.*.*", 100, 3)

    search_results = search_results1 + search_results2

    for result in search_results:
        try:
            domain_search_results.add(extract_domain_name(result['link']))

            if "sitelinks" in result:
                for sitelink in result["sitelinks"]:
                    domain_search_results.add(extract_domain_name(sitelink['link']))
        except KeyError:
            continue

    return domain_search_results

expert_website_researcher_agent_1 = Agent(
        role="Expert Website Researcher",
        goal="Accurately identify the main website of the company {company_name} , which is a part of {main_company}.",
        verbose=True,
        llm=default_llm,
        model_name='gpt-4o',
        allow_delegation=False,
        backstory="""
            You have been a part of {main_company} for many years and have a deep understanding of the company's operations and online presence.
            As a seasoned investigator in the digital realm, you are a skilled web researcher capable of finding accurate company websites using search engines and verifying the information.
            With years of being with {company_name}, you are well known about the ins and outs of this company.
            You know all the websites with copyright same as main website of these.
            You also are expert in google searching and using sites like crunchbase, and Pitch book, etc to find the company details and get the website.
            You are meticulus and organized, and you only provide correct and precise data i.e. websites that you have identified correctly.""",
    )

def process_subsidiary(subsidiary, main_company, sample_expert_website_researcher_output):
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
    
    return results
def main():
    companies = [
        {
            'file': 'Toptal.xlsx',
            'main_company': 'Toptal',
            'output_folder': 'Toptal'
        },
    ]

    sample_expert_website_researcher_output = {
        'subdisiary_name1': [
            'https://www.subdisiary_name1.com',
            'https://www.subdisiary_name1.org'
        ]
    }

    sample_expert_website_researcher_output = json.dumps(sample_expert_website_researcher_output)

    for company in companies:
        start_time = datetime.datetime.now()

        input_file = company['file']
        main_company = company['main_company']
        output_folder = company['output_folder']

        final_results = []

        file_company_list = read_chunks_from_file(input_file)
        print(file_company_list)
        create_result_directory(output_folder)

        official_websites = set()

        with multiprocessing.Pool(processes=10) as pool:
            results = pool.starmap(
                process_subsidiary,
                [(subsidiary, main_company, sample_expert_website_researcher_output) for subsidiary in file_company_list]
            )

        final_results.extend(results)

        data = []
        for result in final_results:
            if isinstance(result, dict):
                for company, urls in result.items():
                    for url in urls:
                        parsed_url = urllib.parse.urlparse(url)
                        scheme = parsed_url.scheme
                        domain_name = parsed_url.netloc

                        if domain_name not in official_websites:
                            official_websites.add(domain_name)
                            full_domain = f"{scheme}://{domain_name}"
                            data.append({'Company Name': company, 'Website URL': full_domain})
            else:
                with open(get_log_file(), 'a') as f:
                    f.write(f"Skipping non-dict result: {result}")
                print(f"Skipping non-dict result: {result}")

        df = pd.DataFrame(data, columns=['Company Name', 'Website URL'])

        df.to_excel('./final_results/' + output_folder + '/website_research_agent' + '.xlsx', engine='openpyxl', index=False)

        print("Data exported")

        df = pd.read_excel('./final_results/' + output_folder + '/website_research_agent' + '.xlsx', engine='openpyxl')

        website_results = set()

        website_urls = df['Website URL'].tolist()

        copyrights = []

        with multiprocessing.Pool(processes=20) as pool:
            results = pool.map(process_website, list(set(website_urls)))
        
        print("Copyrights extracted")

        for domain_name, copyright_info in results:
            website_results.add(domain_name)
            if copyright_info['copyright'] is not None:
                copyrights.append(copyright_info['copyright'])
            else:
                copyrights.append("N/A")

        df['Copyright'] = copyrights

        df.to_excel('./final_results/' + output_folder + '/website_research_agent' + '.xlsx', engine='openpyxl', index=False)

        copyright_results = set()

        df = pd.read_excel('./final_results/' + output_folder + '/website_research_agent' + '.xlsx', engine='openpyxl')
        
        data = df[['Company Name', 'Copyright']]

        data_cleaned = data.dropna(subset=['Copyright'])

        unique_copyrights = data_cleaned.drop_duplicates(subset=['Copyright'])

        with multiprocessing.Pool(processes=20) as pool:
            results = pool.map(process_copyright_research, [row for index, row in unique_copyrights.iterrows()])

        for result in results:
            copyright_results.update(result)

        print(copyright_results)

        df = pd.DataFrame(copyright_results, columns=['Website URL'])

        df = df.to_excel('./final_results/' + output_folder + '/copyright_research_agent' + '.xlsx', engine='openpyxl', index=False)

        print(copyright_results)

        df = pd.read_excel('./final_results/' + output_folder + '/website_research_agent' + '.xlsx', engine='openpyxl')

        website_urls = df['Website URL'].tolist()

        website_urls = set(website_urls)

        website_main_parts = set()
        domain_search_results = set()

        for website in website_urls:
            website_main_parts.add(extract_main_part(website))

        with multiprocessing.Pool(processes=20) as pool:
            results = pool.map(process_domain_research, website_main_parts)

        for result in results:
            domain_search_results.update(result)

        df = pd.DataFrame(domain_search_results, columns=['Website URL'])

        df = df.to_excel('./final_results/' + output_folder + '/domain_search_agent' + '.xlsx', engine='openpyxl', index=False)

        print(domain_search_results)

        combined_final_results = website_results.union(copyright_results).union(domain_search_results)

        df = pd.DataFrame(combined_final_results, columns=['Website URL'])
        df_agentic_output = pd.DataFrame(combined_final_results, columns=['Agentic'])
        df = df.to_excel('./final_results/' + output_folder + '/combined_final_results' + '.xlsx', engine='openpyxl', index=False)

        print(combined_final_results) #agent oupt

        df = pd.read_excel('./final_results/' + output_folder + '/website_research_agent' + '.xlsx', engine='openpyxl')

        website_urls = df['Website URL'].tolist()

        website_urls = set(website_urls)

        link_grabber_results = set()

        with multiprocessing.Pool(processes=20) as pool:
            results = pool.map(get_links, list(website_urls))

            for result in results:
                for link in list(result):
                    link_grabber_results.add(extract_domain_name(link))
        
        df = pd.DataFrame(link_grabber_results, columns=['Website URL'])
        df_link_grabber_output = pd.DataFrame(link_grabber_results, columns=['Link_Grabber'])
        df = df.to_excel('./final_results/' + output_folder + '/link_grabber_agent' + '.xlsx', engine='openpyxl', index=False)

        print(link_grabber_results)

        end_time = datetime.datetime.now()

        print(f"Time taken: {end_time - start_time}")

        with open(get_log_file(), 'a') as f:
            f.write(f"Time taken: {end_time - start_time}")

        print("Processing completed.")

        #code to map link grabber and agentic output into gtd.xlsx file
        # df_original = pd.read_excel(f'final_results/{output_folder}/gtd.xlsx')
        # df_original['Agentic'] = df_agentic_output['Agentic']
        # df_original['Link_Grabber'] = df_link_grabber_output['Link_Grabber']
        # df_original.to_excel(f'final_results/{output_folder}/gtd.xlsx', index=False)

        # print("Data has been written to updated_file.xlsx")

if __name__ == "__main__":
    main()