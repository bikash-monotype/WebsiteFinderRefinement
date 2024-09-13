from helpers import get_scrapegraph_config
from dotenv import load_dotenv
from scrapegraphai.graphs import SmartScraperGraph
from langchain_openai import AzureChatOpenAI
import os
from crewai import Agent, Task, Crew, Process
import multiprocessing
import streamlit as st
from functools import partial
import dill
from helpers import process_worker_function, extract_domain_name, is_working_domain, is_regional_domain_enhanced, translate_text, chunk_list, extract_main_part, social_media_domain_main_part
from tools import search_multiple_page
import json_repair
from helpers import calculate_openai_costs, tokenize_text
import time
import pandas as pd
load_dotenv()

os.environ['AZURE_OPENAI_ENDPOINT'] = os.getenv('AZURE_OPENAI_ENDPOINT')
os.environ['AZURE_OPENAI_CHAT_DEPLOYMENT_NAME'] = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')
os.environ['MODEL_NAME'] = 'Azure'
os.environ['AZURE_OPENAI_API_KEY'] = os.getenv('AZURE_OPENAI_API_KEY')
os.environ['AZURE_OPENAI_API_VERSION'] = '2023-08-01-preview'
os.environ['AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME'] = os.getenv('AZURE_OPENAI_EMBEDDINGS')

model = AzureChatOpenAI(
    azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
    openai_api_version=os.getenv('OPENAI_API_VERSION'),
    api_key=os.getenv('AZURE_OPENAI_API_KEY'),
    model=os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME'),
    temperature=0
)

graph_config = get_scrapegraph_config()

def validate_working_single_domain(log_file_path, domain):
    try :
        is_valid_working_domain = is_working_domain(domain, log_file_path)

        if is_valid_working_domain['is_valid'] is False:
            return {
                'domain': domain,
                'isVisitable': is_valid_working_domain['is_valid'],
                'reason': is_valid_working_domain['reason'],
                'exec_info': None
            }

        prompt = (
            "Check if the site is visitable. If the domain is available for sale, mark it as 'No' and provide a reason. "
            "Also verify if the site is properly reachable (e.g., no errors like 403/404). "
            "Sample output format: {'isVisitable': 'Yes/No', 'reason': 'Explanation if No'}."
        )
        
        smart_scraper_graph = SmartScraperGraph(
            prompt=prompt,
            source=domain,
            config=graph_config,
        )

        result = smart_scraper_graph.run()

        if 'reason' in result:
            reason = result['reason']
        else:
            reason = ''

        graph_exec_info = smart_scraper_graph.get_execution_info()

        return {
            'domain': domain,
            'isVisitable': result['isVisitable'],
            'reason': reason,
            'exec_info': graph_exec_info
        }
    except Exception as e:
        with open(log_file_path['log'], 'a') as f:
            f.write(f"Exception when validating domain using scrapegraph AI: {e}")
        print(f"Exception when validating domain using scrapegraph AI: {e}")
        return {'domain': domain, 'isVisitable': 'No', 'reason': 'Exception when validating domain using scrapegraph AI', 'exec_info': None}
    
def validate_single_correct_domains(log_file_paths, main_company, domain):
    try:
        total_prompt_tokens = 0
        total_completion_tokens = 0

        search_results = search_multiple_page(f"site:{domain} a part of {main_company}?", 10, 1, log_file_path=log_file_paths['log'])

        if len(search_results['all_results']) == 0:
            country_specific_domain = is_regional_domain_enhanced(domain)

            if country_specific_domain is True:
                translate_search_string = translate_text(f"site:{domain} a part of {main_company}?")

                search_results = search_multiple_page(translate_search_string['converted_text'], 10, 1, log_file_path=log_file_paths['log'])

                if len(search_results['all_results']) == 0:
                    return {
                        'results': [domain, 'No'],
                        'llm_usage': {
                            'prompt_tokens': 0,
                            'completion_tokens': 0
                        },
                        'serper_credits': search_results['serper_credits']
                    }
            else:
                return {
                    'results': [domain, 'No'],
                    'llm_usage': {
                        'prompt_tokens': 0,
                        'completion_tokens': 0
                    },
                    'serper_credits': search_results['serper_credits']
                }

        domain_company_validation_researcher = Agent(
            role='Domain Relationship Analyst',
            goal='Validate the relationship between {domain} and {main_company}, assessing whether the domain is officially affiliated with the company. This includes investigating domain ownership, brand association, legal or business affiliations, and any partnerships or acquisitions involving the domain and the company.',
            verbose=True,
            llm=model,
            model_name=os.getenv('AZURE_OPENAI_MODEL_NAME'),
            backstory=(
                """
                As an expert in domain ownership and corporate affiliations, you specialize in identifying the connections between domains and companies. You excel at conducting thorough research using official sources like WHOIS records, company websites, press releases, and legal documents to verify domain ownership and affiliations.
                Your findings are grounded in verifiable data, ensuring that each conclusion about the relationship between a domain and a company is backed by solid, authoritative evidence. You prioritize clarity and accuracy, providing stakeholders with trustworthy information on domain affiliations.
                In your role, you document the findings in a manner that is easy to understand and verify, making sure that all relationships are clearly defined and backed by strong evidence.
                """
            )
        )

        domain_company_validation_task = Task(
            description=(
                """
                Using the search results provided:

                {search_results}

                Determine if the domain "{domain}" is associated with "{main_company}" through one of the following:

                1. Official domain ownership
                2. Entity association
                3. Brand or sub-brand
                4. Acquisition or partnership

                Focus on clear evidence of association, using both exact and partial matches. Consider the context and relationships described.

                **Note:** If the domain is **for sale**, return 'No'

                If the relationship is valid, return 'Yes'. If not, return 'No'

                Only use information from the search results. Avoid assumptions.

                Output format:
                ['{domain}', 'Yes/No']

                **Scoring:**
                - +1 for correct output (based on evidence)
                - -1 for incorrect or speculative output
                """
            ),
            agent=domain_company_validation_researcher,
            expected_output="['{domain}', 'Yes/No']"
        )

        prompt_tokens = tokenize_text(
            f"""
                Domain Relationship Analyst.
                Validate the relationship between {domain} and {main_company}, assessing whether the domain is officially affiliated with the company. This includes investigating domain ownership, brand association, legal or business affiliations, and any partnerships or acquisitions involving the domain and the company.
                As an expert in domain ownership and corporate affiliations, you specialize in identifying the connections between domains and companies. You excel at conducting thorough research using official sources like WHOIS records, company websites, press releases, and legal documents to verify domain ownership and affiliations.
                Your findings are grounded in verifiable data, ensuring that each conclusion about the relationship between a domain and a company is backed by solid, authoritative evidence. You prioritize clarity and accuracy, providing stakeholders with trustworthy information on domain affiliations.
                In your role, you document the findings in a manner that is easy to understand and verify, making sure that all relationships are clearly defined and backed by strong evidence.
                
                Using the search results provided:

                {search_results['all_results']}

                Determine if the domain "{domain}" is associated with "{main_company}" through one of the following:

                1. Official domain ownership
                2. Entity association
                3. Brand or sub-brand
                4. Acquisition or partnership

                Focus on clear evidence of association, using both exact and partial matches. Consider the context and relationships described.

                **Note:** If the domain is **for sale**, return 'No'

                If the relationship is valid, return 'Yes'. If not, return 'No'

                Only use information from the search results. Avoid assumptions.

                Output format:
                ['{domain}', 'Yes/No']

                **Scoring:**
                - +1 for correct output (based on evidence)
                - -1 for incorrect or speculative output
            """
        )

        validation_crew = Crew(
            agents=[domain_company_validation_researcher],
            tasks=[domain_company_validation_task],
            process=Process.sequential,
            verbose=True,
            cache=False
        )

        time.sleep(3)
        
        results = validation_crew.kickoff({
            'domain': domain,
            'main_company': main_company,
            'search_results': search_results['all_results']
        })

        completion_tokens = tokenize_text(f"""
            Thought: I now can give a great answer
            Final Answer: {str(results.raw)}
        """)

        llm_usage = {
            'prompt_tokens': total_prompt_tokens + prompt_tokens,
            'completion_tokens': total_completion_tokens + completion_tokens
        }

        results = json_repair.loads(results.raw)

        with open(log_file_paths['crew_ai'], 'a') as f:
            f.write("\n")
            f.write(f"{domain} validation" + str(llm_usage))

        return {
            'results': results,
            'llm_usage': {
                'prompt_tokens': llm_usage['prompt_tokens'],
                'completion_tokens': llm_usage['completion_tokens']
            },
            'serper_credits': search_results['serper_credits']
        }
    except Exception as e:
        with open(log_file_paths['log'], 'a') as f:
            f.write(f"Exception when validating domain using crew AI: {e}")
        return {'results': [domain, 'No'], 'llm_usage': {'prompt_tokens': 0, 'completion_tokens': 0}, 'serper_credits': 0}

def validate_working_domains(domains, log_file_path):
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_cost_USD = 0.0

    domains = [
        domain if domain.startswith('http://') or domain.startswith('https://') else f'https://{domain}'
        for domain in domains
    ]

    progress_bar = st.progress(0)
    total_domains = len(domains)
    progress_step = 1 / total_domains

    validate_working_domains_with_log = partial(validate_working_single_domain, log_file_path)
    
    serialized_function = dill.dumps(validate_working_domains_with_log)

    with multiprocessing.Pool(processes=20) as pool:
        results = []
        for i, result in enumerate(pool.imap(partial(process_worker_function, serialized_function), domains), 1):
            results.append(result)
            progress_bar.progress(min((i + 1) * progress_step, 1.0))

    invalid_non_working_domains = {}
    valid_working_domains = [] 

    for res in results:
        if res['isVisitable'] != 'Yes':
            invalid_non_working_domains[extract_domain_name(res['domain'])] = res['reason']
        else:
            valid_working_domains.append(extract_domain_name(res['domain']))

        if res['exec_info'] is not None:
            for exec_info in res['exec_info']:
                if exec_info['node_name'] == 'TOTAL RESULT':
                    total_prompt_tokens += exec_info.get('prompt_tokens', 0)
                    total_completion_tokens += exec_info.get('completion_tokens', 0)
                    total_cost_USD += exec_info.get('total_cost_USD', 0.0)

    return {
        'valid_working_domains': valid_working_domains,
        'invalid_non_working_domains': invalid_non_working_domains,
        'total_prompt_tokens': total_prompt_tokens,
        'total_completion_tokens': total_completion_tokens,
        'total_cost_USD': total_cost_USD
    }

def validate_domains(domains, main_company, log_file_path):
    domains = [value for value in domains if not pd.isna(value) and isinstance(value, str) and value != "." and extract_main_part(value) not in social_media_domain_main_part]

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_cost_USD = 0.0

    total_prompt_tokens2 = 0
    total_completion_tokens2 = 0
    total_serper_credits = 0

    st.write('###### Remove incorrect domains')

    progress_bar = st.progress(0)
    total_domains = len(domains)
    progress_step = 1 / total_domains

    validate_single_correct_domains_with_log = partial(validate_single_correct_domains, log_file_path, main_company)

    serialized_function = dill.dumps(validate_single_correct_domains_with_log)

    valid_working_domains = set()
    invalid_non_working_domains = set()

    chunk_size = 10
    results = []

    with multiprocessing.Pool(processes=20) as pool:
        for chunk in chunk_list(domains, chunk_size):
            chunk_results = pool.map(partial(process_worker_function, serialized_function), chunk)
            results.extend(chunk_results)
            progress_bar.progress(min((len(results) + 1) * progress_step, 1.0))

    for res in results:
        if isinstance(res['results'], list) and len(res['results']) >= 2:
            if res['results'][1] != 'Yes':
                invalid_non_working_domains.add(res['results'][0])
            else:
                valid_working_domains.add(res['results'][0])
        else:
            with open(log_file_path['log'], 'a') as f:
                f.write(f"Unexpected format in results: {res['results']}")

        total_prompt_tokens2 += res['llm_usage']['prompt_tokens']
        total_completion_tokens2 += res['llm_usage']['completion_tokens']
        total_serper_credits += res['serper_credits']

    total_cost_USD += calculate_openai_costs(total_prompt_tokens2, total_completion_tokens2)

    with open(log_file_path['llm'], 'a') as f:
        f.write('Remove incorrect domains')
        f.write(f"Total prompt tokens: {total_prompt_tokens2}\n")
        f.write(f"Total completion tokens: {total_completion_tokens2}\n")
        f.write(f"Total cost: {total_cost_USD}\n")

    with open(log_file_path['serper'], 'a') as f:
        f.write('Remove incorrect domains')
        f.write(f"Total Credits: {total_serper_credits}\n")

    # st.write('###### Check for redirection, unreachable and available for sale domains')

    # response = validate_working_domains(list(valid_working_domains_dict.keys()), log_file_path)

    # valid_working_domains = response['valid_working_domains']
    # invalid_non_working_domains = response['invalid_non_working_domains']
    # total_prompt_tokens += response['total_prompt_tokens']
    # total_completion_tokens += response['total_completion_tokens']
    # total_cost_USD += response['total_cost_USD']

    # with open(log_file_path['llm'], 'a') as f:
    #     f.write('Check for redirection, unreachable and available for sale domains')
    #     f.write(f"Total prompt tokens: {response['total_prompt_tokens']}\n")
    #     f.write(f"Total completion tokens: {response['total_completion_tokens']}\n")
    #     f.write(f"Total cost: {response['total_cost_USD']}\n")

    return {
        # 'invalid_non_working_domains': invalid_non_working_domains,
        'final_valid_working_domains': list(valid_working_domains),
        'final_invalid_non_working_domains': list(invalid_non_working_domains),
        'total_prompt_tokens': total_prompt_tokens + total_prompt_tokens2,
        'total_completion_tokens': total_completion_tokens + total_completion_tokens2,
        'total_cost_USD': total_cost_USD,
        'total_serper_credits': total_serper_credits
    }




    

        
    
