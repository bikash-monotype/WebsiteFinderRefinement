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

load_dotenv()

default_llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("OPENAI_API_VERSION"),
    temperature=0,
)

script_dir = os.path.dirname(os.path.abspath(__file__))


def extract_domain_name(url):
    parsed_url = urllib.parse.urlparse(url)
    extracted = tldextract.extract(parsed_url.netloc)
    domain_name = f"{extracted.domain}.{extracted.suffix}"

    return domain_name


def extract_main_part(url):
    parsed_url = urllib.parse.urlparse(url)
    extracted = tldextract.extract(parsed_url.netloc)
    domain_name = extracted.domain

    return domain_name


def read_chunks_from_file(input_file):
    data_file_path = os.path.join(
        script_dir, "data", "subsidiaryfinding-Iteration", input_file
    )

    df = pd.read_excel(data_file_path, header=None, index_col=None)

    # combine the two columns into one df
    df = pd.concat([df[2], df[3]], axis=0)
    df = df.dropna()
    df = df.drop_duplicates()

    # df = df.applymap(lambda x: str(x).replace(",", ""))
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
    with open(file_path, "w") as f:
        f.write("")

    config.log_file = file_path


def get_log_file():
    return config.log_file


def main():
    companies = [
        # {
        #     'file': 'turo_inc.xlsx',
        #     'main_company': 'Turo Inc.',
        #     'output_folder': 'TuroInc'
        # },
        {
            "file": "Delphix Corp..xlsx",
            "main_company": "Delphix Corp.",
            "output_folder": "Delphix Corp.",
        },
    ]

    expert_website_researcher_agent_1 = Agent(
        role="Expert Website Researcher",
        goal="Accurately identify the main website of the company {company_name} , which is a part of {main_company}.",
        verbose=True,
        llm=default_llm,
        model_name="gpt-4o",
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
        "subdisiary_name1": [
            "https://www.subdisiary_name1.com",
            "https://www.subdisiary_name1.org",
        ]
    }

    sample_expert_website_researcher_output = json.dumps(
        sample_expert_website_researcher_output
    )

    for company in companies:
        input_file = company["file"]
        main_company = company["main_company"]
        output_folder = company["output_folder"]

        final_results = []

        file_company_list = read_chunks_from_file(input_file)
        create_result_directory(output_folder)

        print("Starting the process")
        print("File company list: ", file_company_list)

        official_websites = set()

        for subsidiary in file_company_list:
            search_results1 = search_multiple_page(
                f"{subsidiary} a part of {main_company} official website", 10, 1
            )
            search_results2 = search_multiple_page(
                f"{subsidiary} official website", 10, 1
            )
            search_results3 = search_multiple_page(f"{subsidiary}", 10, 1)

            search_results = json.dumps(
                search_results1 + search_results2 + search_results3
            )

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
                verbose=1,
            )

            results = expert_website_researcher_crew_1.kickoff(
                inputs={
                    "company_name": subsidiary,
                    "main_company": main_company,
                    "search_results": search_results,
                    "sample_expert_website_researcher_output": sample_expert_website_researcher_output,
                }
            )
            results = json_repair.loads(results.raw)

            final_results.append(results)

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
                            data.append(
                                {"Company Name": company, "Website URL": full_domain}
                            )
            else:
                with open(get_log_file(), "a") as f:
                    f.write(f"Skipping non-dict result: {result}")
                print(f"Skipping non-dict result: {result}")

        df = pd.DataFrame(data, columns=["Company Name", "Website URL"])

        df.to_excel(
            "./final_results/" + output_folder + "/website_research_agent" + ".xlsx",
            engine="openpyxl",
            index=False,
        )

        print("Data exported")

        df = pd.read_excel(
            "./final_results/" + output_folder + "/website_research_agent" + ".xlsx",
            engine="openpyxl",
        )

        website_results = set()

        website_urls = df["Website URL"].tolist()
        copyrights = []

        for website in website_urls:
            website_results.add(extract_domain_name(website))
            copyright = get_copyright(website)
            copyrights.append(copyright["copyright"])

        df["Copyright"] = copyrights

        df.to_excel(
            "./final_results/" + output_folder + "/website_research_agent" + ".xlsx",
            engine="openpyxl",
            index=False,
        )

        df = pd.read_excel(
            "./final_results/" + output_folder + "/website_research_agent" + ".xlsx",
            engine="openpyxl",
        )

        copyrights = df["Copyright"].tolist()
        copyrights = set(copyrights)

        copyright_results = set()

        for copyright in list(copyrights):
            copyright_result = search_multiple_page(
                f'"{copyright}" -linkedin -quora -instagram -youtube -facebook -twitter -pinterest -snapchat -github -whatsapp -tiktok -reddit -news -blog -articles -forums -pdf -sec.gov -x.com -amazon -vimeo',
                100,
                3,
            )

            for result in copyright_result:
                try:
                    copyright_results.add(extract_domain_name(result["link"]))

                    if "sitelinks" in result:
                        copyright_results.add(extract_domain_name(result["link"]))
                except KeyError:
                    continue

        df = pd.DataFrame(copyright_results, columns=["Website URL"])

        df = df.to_excel(
            "./final_results/" + output_folder + "/copyright_research_agent" + ".xlsx",
            engine="openpyxl",
            index=False,
        )

        print(copyright_results)

        df = pd.read_excel(
            "./final_results/" + output_folder + "/website_research_agent" + ".xlsx",
            engine="openpyxl",
        )

        website_urls = df["Website URL"].tolist()

        website_urls = set(website_urls)

        website_main_parts = set()
        domain_search_results = set()

        for website in website_urls:
            website_main_parts.add(extract_main_part(website))

        for main_part in website_main_parts:
            search_results1 = search_multiple_page(f"site:{main_part}.*", 100, 3)
            search_results2 = search_multiple_page(f"site:{main_part}.*.*", 100, 3)

            search_results = search_results1 + search_results2

            for result in search_results:
                try:
                    domain_search_results.add(extract_domain_name(result["link"]))

                    if "sitelinks" in result:
                        domain_search_results.add(extract_domain_name(result["link"]))
                except KeyError:
                    continue

        df = pd.DataFrame(domain_search_results, columns=["Website URL"])

        df = df.to_excel(
            "./final_results/" + output_folder + "/domain_search_agent" + ".xlsx",
            engine="openpyxl",
            index=False,
        )

        print(domain_search_results)

        combined_final_results = website_results.union(copyright_results).union(
            domain_search_results
        )

        df = pd.DataFrame(combined_final_results, columns=["Website URL"])

        df = df.to_excel(
            "./final_results/" + output_folder + "/combined_final_results" + ".xlsx",
            engine="openpyxl",
            index=False,
        )

        print(combined_final_results)


if __name__ == "__main__":
    main()
