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
from helpers import process_worker_function, extract_domain_name, is_working_domain
from tools import search_multiple_page
import json_repair
from helpers import calculate_openai_costs

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
    search_results = search_multiple_page(f"site:{domain} a part of {main_company}?", 10, 1, log_file_path=log_file_paths['log'])

    if len(search_results['all_results']) == 0:
        return {
            'results': [domain, 'No', 'No search results'],
            'llm_usage': {
                'prompt_tokens': 0,
                'completion_tokens': 0
            },
            'serper_credits': search_results['serper_credits']
        }

    domain_company_validation_task = Task(
        description=(
            """
            Using the search results provided:

            {search_results}

            Determine if {domain} is related to {main_company} as an official domain, brand, sub-brand, entity, acquisition, or a significant partnership. 
            Focus exclusively on the information relevant to domain ownership and company affiliations. Be meticulous in validating the source of each piece of data. If no definitive information is available, specify 'N/A'. Incorrect or speculative entries will result in penalties.
            It is critical to cite the exact source that confirms the nature of the relationship. Ensure that all responses adhere to the expected output format to avoid penalties.
            """
        ),
        agent=domain_company_validation_researcher,
        expected_output="['{domain}', 'Yes/No', 'Source URL']"
    )

    validation_crew = Crew(
        agents=[domain_company_validation_researcher],
        tasks=[domain_company_validation_task],
        process=Process.sequential,
        verbose=True
    )

    results = validation_crew.kickoff({
        'domain': domain,
        'main_company': main_company,
        'search_results': search_results['all_results']
    })

    results = json_repair.loads(results.raw)
    llm_usage = validation_crew.calculate_usage_metrics()

    return {
        'results': results,
        'llm_usage': {
            'prompt_tokens': llm_usage.prompt_tokens,
            'completion_tokens': llm_usage.completion_tokens
        },
        'serper_credits': search_results['serper_credits']
    }

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
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_cost_USD = 0.0

    total_prompt_tokens2 = 0
    total_completion_tokens2 = 0
    total_serper_credits = 0

    response = validate_working_domains(domains, log_file_path)

    valid_working_domains = response['valid_working_domains']
    invalid_non_working_domains = response['invalid_non_working_domains']
    total_prompt_tokens += response['total_prompt_tokens']
    total_completion_tokens += response['total_completion_tokens']
    total_cost_USD += response['total_cost_USD']

    st.write('###### Remove incorrect domains')

    progress_bar = st.progress(0)
    total_domains = len(valid_working_domains)
    progress_step = 1 / total_domains

    validate_single_correct_domains_with_log = partial(validate_single_correct_domains, log_file_path, main_company)

    serialized_function = dill.dumps(validate_single_correct_domains_with_log)

    valid_working_domains_dict = {}
    invalid_non_working_domains_dict = {}

    with multiprocessing.Pool(processes=20) as pool:
        results = []
        for i, result in enumerate(pool.imap(partial(process_worker_function, serialized_function), valid_working_domains), 1):
            results.append(result)
            progress_bar.progress(min((i + 1) * progress_step, 1.0))

    for res in results:
        if len(res['results']) >= 3:
            if res['results'][1] != 'Yes':
                invalid_non_working_domains_dict[res['results'][0]] = res['results'][2]
            else:
                valid_working_domains_dict[res['results'][0]] = res['results'][2]
        else:
            with open(log_file_path['log'], 'a') as f:
                f.write(f"Unexpected format in results: {res['results']}")

        total_prompt_tokens2 += res['llm_usage']['prompt_tokens']
        total_completion_tokens2 += res['llm_usage']['completion_tokens']
        total_serper_credits += res['serper_credits']

    total_cost_USD += calculate_openai_costs(total_prompt_tokens2, total_completion_tokens2)

    return {
        'invalid_non_working_domains': invalid_non_working_domains,
        'final_valid_working_domains_dict': valid_working_domains_dict,
        'final_invalid_non_working_domains_dict': invalid_non_working_domains_dict,
        'total_prompt_tokens': total_prompt_tokens + total_prompt_tokens2,
        'total_completion_tokens': total_completion_tokens + total_completion_tokens2,
        'total_cost_USD': total_cost_USD,
        'total_serper_credits': total_serper_credits
    }




    

        
    
