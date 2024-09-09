import multiprocessing
import datetime
import urllib.parse
from crewai import Agent, Task, Crew, Process
from langchain_openai import AzureChatOpenAI
from tools import search_multiple_page
import os
import json_repair
import streamlit as st
import json
from functools import partial
from helpers import extract_domain_name
from copyright import get_copyright
import re
from helpers import extract_year, extract_main_part, get_links, process_worker_function
import pandas as pd
import dill
from dotenv import load_dotenv

load_dotenv()

default_llm = AzureChatOpenAI(
    azure_endpoint='',
    model=os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME'),
    openai_api_version=os.getenv('OPENAI_API_VERSION'),
    temperature=0
)

expert_website_researcher_agent_1 = Agent(
    role="Expert Website Researcher",
    goal="Accurately identify the main website of the company {company_name} , which is a part of {main_company}.",
    verbose=True,
    llm=default_llm,
    model_name=os.getenv('AZURE_OPENAI_MODEL_NAME'),
    allow_delegation=False,
    backstory="""
        You have been a part of {main_company} for many years and have a deep understanding of the company's operations and online presence.
        As a seasoned investigator in the digital realm, you are a skilled web researcher capable of finding accurate company websites using search engines and verifying the information.
        With years of being with {company_name}, you are well known about the ins and outs of this company.
        You know all the websites with copyright same as main website of these.
        You also are expert in google searching and using sites like crunchbase, and Pitch book, etc to find the company details and get the website.
        You are meticulus and organized, and you only provide correct and precise data i.e. websites that you have identified correctly.""",
)

def process_single_website(website, log_file_path):
    result = get_copyright(website, log_file_path)
    return {
        'website': website,
        'result': result
    }

def process_website_and_get_copyrights(websites, log_file_paths):
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_cost_USD = 0.0

    progress_bar = st.progress(0)
    total_websites = len(websites)
    progress_step = 1 / total_websites
    
    process_website_with_log_file = partial(process_single_website, log_file_path=log_file_paths)

    serialized_function = dill.dumps(process_website_with_log_file)
    
    with multiprocessing.Pool(processes=20) as pool:
        results = []
        for i, result in enumerate(pool.starmap(process_worker_function, [(serialized_function, part) for part in list(websites)])):
            results.append(result)
            progress_bar.progress((i + 1) * progress_step)

    copyrights_dict = {}

    for data in results:
        website = data['website']
        result = data['result']

        copyright_info = result['result']
        if copyright_info['copyright'] is not None:
            copyrights_dict[website] = copyright_info['copyright']
        else:
            copyrights_dict[website] = "N/A"

        for exec_info in result['exec_info']:
            if exec_info['node_name'] == 'TOTAL RESULT':
                total_prompt_tokens += exec_info.get('prompt_tokens', 0)
                total_completion_tokens += exec_info.get('completion_tokens', 0)
                total_cost_USD += exec_info.get('total_cost_USD', 0.0)

    return {
        'copyrights': copyrights_dict,
        'llm_usage': {
            'prompt_tokens': total_prompt_tokens,
            'completion_tokens': total_completion_tokens,
            'total_cost_USD': total_cost_USD
        }
    }

def process_subsidiary(subsidiary, main_company, sample_expert_website_researcher_output, log_file_paths):
    search_results1 = search_multiple_page(f"{subsidiary} a part of {main_company} official website", 10, 1, log_file_paths['log'])
    search_results2 = search_multiple_page(f"{subsidiary} official website", 10, 1, log_file_paths['log'])
    search_results3 = search_multiple_page(f"{subsidiary}", 10, 1, log_file_paths['log'])

    total_serper_credits = search_results1['serper_credits'] + search_results2['serper_credits'] + search_results3['serper_credits']

    search_results = json.dumps(search_results1['all_results'] + search_results2['all_results'] + search_results3['all_results'])

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
                6. Exclude unrelated third-party profiles (e.g., Bloomberg, Meta, LinkedIn, Pitchbook, App store, Wikipedia, Encyclopedia) unless they are the primary online presence of the company.
                7. List all identified websites in a clear and organized manner.

                Sample Output:
                {sample_expert_website_researcher_output}

                Important notes:
                - Every set of rules and steps mentioned above must be followed to get the required results.
                - Do not provide any other texts or information in the output as it will not work with the further process.
                - Do not include ``` or any other such characters in the output.
            """
        ),
        agent=expert_website_researcher_agent_1,
        expected_output="All possible official website of the company. {company_name}",
    )

    expert_website_researcher_crew_1 = Crew(
        agents=[expert_website_researcher_agent_1],
        tasks=[expert_website_researcher_task_1],
        process=Process.sequential,
        verbose=1
    )
    
    results = expert_website_researcher_crew_1.kickoff(inputs={"company_name": subsidiary, "main_company": main_company, "search_results": search_results, "sample_expert_website_researcher_output": sample_expert_website_researcher_output})
    print(results.raw)
    results = json_repair.loads(results.raw)

    llm_usage = expert_website_researcher_crew_1.calculate_usage_metrics()
    
    return {
        'websites': results,
        'llm_usage': {
            'prompt_tokens': llm_usage.prompt_tokens,
            'completion_tokens': llm_usage.completion_tokens
        },
        'serper_credits': total_serper_credits
    }

def process_single_domain_research(main_part, log_file_paths):
    domain_search_results = set()
    search_results1 = search_multiple_page(f"site:{main_part}.*", 100, 3, log_file_paths['log'])
    search_results2 = search_multiple_page(f"site:{main_part}.*.*", 100, 3, log_file_paths['log'])

    search_results = search_results1['all_results'] + search_results2['all_results']

    for result in search_results:
        try:
            domain_search_results.add(extract_domain_name(result['link']))

            if "sitelinks" in result:
                for sitelink in result["sitelinks"]:
                    domain_search_results.add(extract_domain_name(sitelink['link']))
        except KeyError:
            continue

    total_seper_credits = search_results1['serper_credits'] + search_results2['serper_credits']

    return {
        'domain_search_results': domain_search_results,
        'serper_credits': total_seper_credits
    }

def process_domain_research(website_urls, log_file_paths):
    website_urls = set(website_urls)

    total_serper_credits = 0

    website_main_parts = set()
    domain_search_results = set()

    for website in website_urls:
        website_main_parts.add(extract_main_part(website))

    progress_bar = st.progress(0)
    total_domains = len(website_main_parts)
    progress_step = 1 / total_domains

    process_domain_research_with_log_file = partial(process_single_domain_research, log_file_paths=log_file_paths)

    serialized_function = dill.dumps(process_domain_research_with_log_file)
    
    with multiprocessing.Pool(processes=20) as pool:
        results = []
        for i, result in enumerate(pool.starmap(process_worker_function, [(serialized_function, part) for part in website_main_parts])):
            results.append(result)
            progress_bar.progress((i + 1) * progress_step)

    for result in results:
        total_serper_credits += result['serper_credits']
        domain_search_results.update(result['domain_search_results'])

    return {
        'domain_search_results': domain_search_results,
        'serper_credits': total_serper_credits
    }

def process_link_grabber(website_urls, log_file_paths):
    website_urls = set(website_urls)

    progress_bar = st.progress(0)
    total_urls = len(website_urls)
    progress_step = 1 / total_urls

    link_grabber_results = set()

    get_links_with_log_file = partial(get_links, log_file_path=log_file_paths['log'])

    serialized_function = dill.dumps(get_links_with_log_file)

    with multiprocessing.Pool(processes=20) as pool:
        results = []
        for i, result in enumerate(pool.starmap(process_worker_function, [(serialized_function, url) for url in list(website_urls)])):
            results.append(result)
            progress_bar.progress((i + 1) * progress_step)

    for result in results:
        for link in list(result):
            link_grabber_results.add(extract_domain_name(link))

    return link_grabber_results

def process_single_copyright_research(row, log_file_paths):
    company_name = row['Company Name']
    copyright = row['Copyright']
    copyright_results = set()

    year = extract_year(copyright)

    copyright_result1 = search_multiple_page(
        f"'{copyright}' -linkedin -quora -instagram -youtube -facebook -twitter -pinterest -snapchat -github -whatsapp -tiktok -reddit -x.com -amazon -vimeo", 100, 3, log_file_paths['log'])
    
    if year is not None:
        copyright_result2 = search_multiple_page(
            f"'© {year} {company_name}' -linkedin -quora -instagram -youtube -facebook -twitter -pinterest -snapchat -github -whatsapp -tiktok -reddit -x.com -amazon -vimeo", 100, 3, log_file_paths['log'])
    else:
        copyright_result2 = {
            'all_results': [],
            'serper_credits': 0
        }

    words = copyright.split()
    filtered_words = [word for word in words if not re.search(r'\b(group|ltd)\b', word, re.IGNORECASE)]
    filtered_copyright = ' '.join(filtered_words)
    filtered_copyright_ran_already = False

    if filtered_copyright != copyright:
        filtered_copyright_ran_already = True
        copyright_result3 = search_multiple_page(
            f"'{filtered_copyright}' -linkedin -quora -instagram -youtube -facebook -twitter -pinterest -snapchat -github -whatsapp -tiktok -reddit -x.com -amazon -vimeo", 100, 3, log_file_paths['log'])
    else:
        copyright_result3 = {
            'all_results': [],
            'serper_credits': 0
        }

    copyright_result4 = {
        'all_results': [],
        'serper_credits': 0
    }

    if ( year is not None and filtered_copyright != ("© "+ year + " " + company_name)):
        if filtered_copyright_ran_already is False:
            copyright_result4 = search_multiple_page(
                f"'{filtered_copyright}' -linkedin -quora -instagram -youtube -facebook -twitter -pinterest -snapchat -github -whatsapp -tiktok -reddit -x.com -amazon -vimeo", 100, 3, log_file_paths['log'])

    copyright_result = copyright_result1['all_results'] + copyright_result2['all_results'] + copyright_result3['all_results'] + copyright_result4['all_results']

    for result in copyright_result:
        try:
            copyright_results.add(extract_domain_name(result['link']))

            if "sitelinks" in result:
                for sitelink in result["sitelinks"]:
                    copyright_results.add(extract_domain_name(sitelink['link']))
        except KeyError:
            continue

    total_serper_credits = copyright_result1['serper_credits'] + copyright_result2['serper_credits'] + copyright_result3['serper_credits'] + copyright_result4['serper_credits']

    return {
        'copyright_results': copyright_results,
        'serper_credits': total_serper_credits
    }

def process_copyright_research(unique_copyrights, log_file_paths):
    copyright_results = set()

    total_serper_credits = 0

    progress_bar = st.progress(0)
    total_copyrights = unique_copyrights.shape[0]
    progress_step = 1 / total_copyrights

    process_copyright_research_with_log_file = partial(process_single_copyright_research, log_file_paths=log_file_paths)

    serialized_function = dill.dumps(process_copyright_research_with_log_file)
    
    with multiprocessing.Pool(processes=20) as pool:
        results = []
        for i, result in enumerate(pool.starmap(process_worker_function, [(serialized_function, row) for index, row in unique_copyrights.iterrows()])):
            results.append(result)
            progress_bar.progress((i + 1) * progress_step)

    for result in results:
        total_serper_credits += result['serper_credits']
        copyright_results.update(result['copyright_results'])

    return {
        'copyright_results': copyright_results,
        'serper_credits': total_serper_credits
    }

def get_official_websites(file_company_list, main_company, log_file_paths):
    sample_expert_website_researcher_output = {
        'subdisiary_name1': [
            'https://www.subdisiary_name1.com',
            'https://www.subdisiary_name1.org'
        ]
    }

    final_results = []

    progress_bar = st.progress(0)
    total_subsidiaries = len(file_company_list)
    progress_step = 1 / total_subsidiaries

    with multiprocessing.Pool(processes=10) as pool:
        results = []
        for i, result in enumerate(pool.starmap(
                process_subsidiary,
                [(subsidiary, main_company, sample_expert_website_researcher_output, log_file_paths) for subsidiary in file_company_list]
            )):
            results.append(result)
            progress_bar.progress((i + 1) * progress_step)

    final_results.extend(results)

    total_website_researcher_prompt_tokens = 0
    total_website_researcher_completion_tokens = 0
    total_website_researcher_serper_credits = 0

    data = []
    for result in final_results:
        if isinstance(result, dict):
            if 'websites' in result and 'llm_usage' in result:
                websites = result['websites']
                llm_usage = result['llm_usage']
                
                total_website_researcher_prompt_tokens += llm_usage['prompt_tokens']
                total_website_researcher_completion_tokens += llm_usage['completion_tokens']
                total_website_researcher_serper_credits += result['serper_credits']
                
                for company, urls in websites.items():
                    for url in urls:
                        data.append({'Company Name': company, 'Website URL': url})
            else:
                with open(log_file_paths['log'], 'a') as f:
                    f.write(f"Skipping result with missing 'websites' or 'llm_usage' key: {result}")
        else:
            with open(log_file_paths['log'], 'a') as f:
                f.write(f"Skipping non-dict result from expert website researcher: {result}")
    return {
        'data': data,
        'llm_usage': {
            'prompt_tokens': total_website_researcher_prompt_tokens,
            'completion_tokens': total_website_researcher_completion_tokens
        },
        'serper_credits': total_website_researcher_serper_credits
    }

