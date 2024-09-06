from textwrap import dedent
import pandas as pd
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import json_repair
import os
from dotenv import load_dotenv
import json
import streamlit as st

load_dotenv()

model = AzureChatOpenAI(
    azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
    azure_deployment=os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME'),
    openai_api_version=os.getenv('OPENAI_API_VERSION'),
    temperature=0
)

def pad_list(lst, length):
    return lst + [None] * (length - len(lst))

def validate_company_structure(agentsOutput, company_name):
    agentsOutputList = list(agentsOutput)

    llm_usage = {
        'prompt_tokens': 0,
        'completion_tokens': 0
    }

    if len(agentsOutputList) == 0:
        st.write('No company structure output.')
    else:
        correct_agentsOutputList = []
        error_agentsOutputList = []

        sample_output_json = json.dumps({
            'Company1': True,
            'Company2': True,
            'Company3': False,
            'Company4': True,
            'Company5': False,
        })

        progress_bar = st.progress(0)
        total_chunks = (len(agentsOutputList) + 99) // 100
        progress_step = 1 / total_chunks

        for i in range(0, len(agentsOutputList), 100):
            chunk = agentsOutputList[i:i+100]

            data = {
                'main_company': company_name,
                'companies': chunk
            }
            
            messages = [
                SystemMessage(content="""As a Expert Company Researcher at a large organization, your task is to analyze each companies in the provided company list and provide whether each company is a subsidiaries/brands/sub-brands/acquisitions/trusts/entities/global operations/charitable organizations/companies with more than 50% partnership of {company_name}."""),
                HumanMessage(content=f"""
                    You will be given a Python dictionary with two keys: 'main_company' and 'companies'. The 'main_company' key represents the name of the main company, and the 'companies' key contains a list of company names. 
                    Your task is to determine whether each company in the list is a subsidiary, brand, sub-brand, acquisitions, trusts, entities, global operations, charitable organizations and companies with more than 50% partnership of the main company. 

                    This task should be completed without using algorithms or models, relying solely on your expertise and experience.

                    Here is the data you'll be working with: {data}

                    Sample Output Format:
                    {sample_output_json}

                    Do not provide any explanations or logic for your output. The keys in your output must match the values in the provided list. 
                    Ensure your output follows the exact format provided in the sample. Failure to adhere to the instructions may result in penalties, and your output will be invalid if it cannot be parsed.
                """),
            ]

            response = model.invoke(messages)

            response_json = json_repair.loads(response.content)

            llm_usage['prompt_tokens'] += response.usage_metadata['input_tokens']
            llm_usage['completion_tokens'] += response.usage_metadata['output_tokens']

            true_keys = [key for key, value in response_json.items() if value is True]
            false_keys = [key for key, value in response_json.items() if value is False]

            correct_agentsOutputList.extend(true_keys)
            error_agentsOutputList.extend(false_keys)

            progress_bar.progress((i // 100 + 1) * progress_step)

        accuracy = (len(correct_agentsOutputList)) / (len(agentsOutputList)) * 100

        error = (len(error_agentsOutputList) / len(agentsOutputList)) * 100

        max_length = max(len(agentsOutputList), len(correct_agentsOutputList), len(error_agentsOutputList))

        agentsOutputList = pad_list(agentsOutputList, max_length)
        correct_agentsOutputList = pad_list(correct_agentsOutputList, max_length)
        error_agentsOutputList = pad_list(error_agentsOutputList, max_length)

        return {
            'agentsOutputList': agentsOutputList,
            'correct_agentsOutputList': correct_agentsOutputList,
            'error_agentsOutputList': error_agentsOutputList,
            'accuracy': accuracy,
            'error': error,
            'llm_usage': llm_usage
        }

