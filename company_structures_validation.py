from langchain_openai import AzureChatOpenAI
import json_repair
import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
import streamlit as st
import multiprocessing
from helpers import process_worker_function, calculate_openai_costs, pad_list
import dill
from functools import partial
from tools import search_multiple_page

load_dotenv()

os.environ["SERPER_API_KEY"] = os.getenv('SERPER_API_KEY')

model = AzureChatOpenAI(
    azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
    openai_api_version=os.getenv('OPENAI_API_VERSION'),
    api_key=os.getenv('AZURE_OPENAI_API_KEY'),
    model=os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME'),
    temperature=0
)

company_structures_validation_researcher = Agent(
    role='Corporate Relationship Analyst',
    goal='Evaluate the relationship between {subsidiary} and {main_company} across multiple dimensions, including but not limited to subsidiary status, branding, acquisitions, partnerships, and organizational affiliations.',
    verbose=True,
    llm=model,
    model_name=os.getenv('AZURE_OPENAI_MODEL_NAME'),
    max_iter=5,
    backstory=(
        """
        As an expert in corporate affiliations and structures, you have developed an ability to discern complex corporate relationships using official documentation and reliable sources. You have a proven track record of accurately identifying the nature of business relationships, ensuring that all findings are grounded in verifiable data.
        Your research is always directed towards official corporate sites or authoritative government records. This ensures that the information you gather is both relevant and legally sound. Each claim about a relationship is backed by solid evidence from trusted sources, eliminating any room for ambiguity or error.
        Your role involves not only identifying these relationships but also documenting them in a way that can be easily understood and verified by stakeholders, ensuring clarity and accountability in corporate governance.
        """
    )
)

def process_single_company_structure_validation(main_company, subsidiary, log_file_paths):
    search_results = search_multiple_page(f"is {subsidiary} a part of {main_company}?", 10, 1, log_file_path=log_file_paths['log'])

    company_structures_validation_task = Task(
        description=(
            """
            Using the search results provided:

            {search_results}

            Determine if {subsidiary} is related to {main_company} as a subsidiary, brand, sub-brand, acquisition, trust, entity, global operation, charitable organization, or holds a significant partnership (>50% ownership).
            Focus exclusively on the information above. Be meticulous in validating the source of each piece of data. If no definitive information is available, specify 'N/A'. Incorrect or speculative entries will result in penalties.
            It is critical to cite the exact source that confirms the nature of the relationship. Ensure that all responses adhere to the expected output format to avoid penalties.
            """
        ),
        agent=company_structures_validation_researcher,
        expected_output="['{subsidiary}', 'Yes/No', 'Source URL']"  # Specify that a URL is expected for source verification
    )

    validation_crew = Crew(
        agents=[company_structures_validation_researcher],
        tasks=[company_structures_validation_task],
        process=Process.sequential,
        verbose=True
    )

    results = validation_crew.kickoff({
        'subsidiary': subsidiary,
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

def validate_company_structure(agentsOutput, company_name, log_file_paths):
    agentsOutputList = list(agentsOutput)

    correct_agentsOutputDict = {}
    error_agentsOutputDict = {}

    llm_usage = {
        'prompt_tokens': 0,
        'completion_tokens': 0
    }

    total_serper_credits = 0

    if len(agentsOutputList) == 0:
        st.write('No company structure output.')
    else:
        progress_bar = st.progress(0)
        total_chunks = (len(agentsOutputList) + 99) // 100
        progress_step = 1 / total_chunks

        process_company_structure_validation_with_log_file = partial(process_single_company_structure_validation, company_name, log_file_paths=log_file_paths)

        serialized_function = dill.dumps(process_company_structure_validation_with_log_file)

        with multiprocessing.Pool(processes=20) as pool:
            results = []
            for i, result in enumerate(pool.starmap(process_worker_function, [(serialized_function, part) for part in list(agentsOutputList)])):
                results.append(result)
                progress_bar.progress(min((i + 1) * progress_step, 1.0))

        for res in results:
            if 'results' in res and len(res['results']) >= 3:
                if res['results'][1] == 'Yes':
                    correct_agentsOutputDict[res['results'][0]] = res['results'][2]
                else:
                    error_agentsOutputDict[res['results'][0]] = res['results'][2]
            else:
                with open(log_file_paths['log'], 'a') as f:
                    f.write(f"Skipping non-dict result from expert website researcher: {str(res)}\n")

            llm_usage['prompt_tokens'] += res['llm_usage']['prompt_tokens']
            llm_usage['completion_tokens'] += res['llm_usage']['completion_tokens']
            total_serper_credits += res['serper_credits']

        correct_agentsOutputList = list(correct_agentsOutputDict.keys())
        error_agentsOutputList = list(error_agentsOutputDict.keys())
        correct_agentsOutputListReference = list(correct_agentsOutputDict.values())
        error_agentsOutputListReference = list(error_agentsOutputDict.values())

        accuracy = (len(correct_agentsOutputList)) / (len(agentsOutputList)) * 100
        error = (len(error_agentsOutputList) / len(agentsOutputList)) * 100

        total_llm_costs = calculate_openai_costs(llm_usage['prompt_tokens'], llm_usage['completion_tokens'])

        max_length = max(len(agentsOutputList), len(correct_agentsOutputList), len(error_agentsOutputList))

        agentsOutputList = pad_list(agentsOutputList, max_length)
        correct_agentsOutputList = pad_list(correct_agentsOutputList, max_length)
        correct_agentsOutputListReference = pad_list(correct_agentsOutputListReference, max_length)
        error_agentsOutputList = pad_list(error_agentsOutputList, max_length)
        error_agentsOutputListReference = pad_list(error_agentsOutputListReference, max_length)

        return {
            'agentsOutputList': agentsOutputList,
            'correct_agentsOutputList': correct_agentsOutputList,
            'correct_agentsOutputListReference': correct_agentsOutputListReference,
            'error_agentsOutputList': error_agentsOutputList,
            'error_agentsOutputListReference': error_agentsOutputListReference,
            'accuracy': accuracy,
            'error': error,
            'llm_usage': {
                'prompt_tokens': llm_usage['prompt_tokens'],
                'completion_tokens': llm_usage['completion_tokens'],
                'total_llm_costs': total_llm_costs
            },
            'serper_credits': total_serper_credits
        }

