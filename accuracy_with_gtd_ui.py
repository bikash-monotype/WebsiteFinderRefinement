import streamlit as st
import pandas as pd
from datetime import datetime
from helpers import create_result_directory, extract_domain_name, pad_list, social_media_domain_main_part, extract_main_part
import os
from company_websites_validation import validate_agentsOutput_domains, validate_working_domains, validate_linkgrabber_domains
from company_websites import process_single_website

import json
import re

def clean_url(url):
    url = re.sub(r'^https?://', '', url)
    url = re.sub(r'^www\.', '', url)
    return f"https://{url}"

st.title("Accuracy check with GTD")

uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])

link_grabber_file = st.file_uploader("Choose a file for link grabber", type=["json"])

company_name = st.text_input("Enter company name")
company_website = st.text_input("Enter company website")

if uploaded_file is not None and link_grabber_file is not None and company_name is not None:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    if link_grabber_file.name.endswith('.json'):
        link_grabber_data = json.load(link_grabber_file)
    
    if st.button("Process File"):
        print("Processing file...")
        start_time = datetime.now()
        gtd = df['GTD'].tolist()
        agentsOutput = df['AgentsOutput'].tolist()

        gtd = [value for value in gtd if not pd.isna(value)]
        agentsOutput = [value for value in agentsOutput if not pd.isna(value) and isinstance(value, str) and value != "." and extract_main_part(value) not in social_media_domain_main_part]

        gtd = [extract_domain_name(url) for url in gtd]
        gtd = set(gtd)
        gtd = list(gtd)

        agentsOutput = set(agentsOutput)
        agentsOutput = list(agentsOutput)

        folder_name = datetime.now().strftime("%Y%m%d%H%M%S")
        result_directory = f"{company_name}_{folder_name}"
        final_results_directory = f"validation/{result_directory}"
        log_file_paths = create_result_directory(result_directory, 'validation')

        whole_process_prompt_tokens = 0
        whole_process_completion_tokens = 0
        whole_process_llm_costs = 0
        whole_process_serper_credits = 0

        st.markdown(f"""
            <style>
                .output-folder {{
                    color: green;
                    font-weight: bold;
                }}
            </style>
            <div class="output-folder">
                The output folder is: {result_directory}. Please remember this for future reference.
            </div>
        """, unsafe_allow_html=True)

        copyright = process_single_website(company_website, log_file_paths)
        copyright_text = copyright['result']['result']['copyright']

        with open(log_file_paths['log'], 'a') as f:
            f.write("\n\n")
            f.write(f"Get Copyright of main domain: {copyright_text}")

        if copyright['result']['exec_info'] is not None:
            for exec_info in copyright['result']['exec_info']:
                if exec_info['node_name'] == 'TOTAL RESULT':
                    whole_process_prompt_tokens += exec_info.get('prompt_tokens', 0)
                    whole_process_completion_tokens += exec_info.get('completion_tokens', 0)
                    whole_process_llm_costs += exec_info.get('total_cost_USD', 0.0)

        company_domain = clean_url(company_website)
        response = validate_agentsOutput_domains(agentsOutput, company_name, company_domain, copyright_text, log_file_paths)

        export_df = pd.DataFrame(response['agentsOutput_validation_AI_responses'], columns=['Domain', 'Ownership Not Clear', 'AI Response', 'Url', 'Reason'])

        export_df.to_excel(os.path.join(final_results_directory, 'agentsOutput_validation_AI_responses.xlsx'), index=False, header=True)

        export_df = pd.DataFrame({
            'Domains': response['invalid_non_working_domains'],
        })

        export_df.to_excel(os.path.join(final_results_directory, 'invalid_non_working_domains_excluding_linkgrabber.xlsx'), index=False, header=True)

        filtered_link_grabber_data = {}

        for data in link_grabber_data:
            for main_domain, domains in data.items():
                if main_domain in response['valid_working_domains']:
                    for domain in domains:
                        if domain not in response['valid_working_domains']:
                            if main_domain in filtered_link_grabber_data:
                                filtered_link_grabber_data[main_domain].append(domain)
                            else:
                                filtered_link_grabber_data[main_domain] = [domain]

        if len(filtered_link_grabber_data) != 0:
            response2 = validate_linkgrabber_domains(company_name, filtered_link_grabber_data, log_file_paths)

            export_df = pd.DataFrame(response2['link_grabber_validation_AI_responses'], columns=['Main Domain', 'Domain', 'AI Response', 'Reason', 'Url'])

            export_df.to_excel(os.path.join(final_results_directory, 'link_grabber_validation_AI_responses.xlsx'), index=False, header=True)

            export_df = pd.DataFrame({
                'Domains': response2['invalid_non_working_domains'],
            })

            export_df.to_excel(os.path.join(final_results_directory, 'invalid_non_working_domains_of_linkgrabber.xlsx'), index=False, header=True)

            valid_domains = set(response['valid_working_domains'] + response2['valid_working_domains'])

            whole_process_prompt_tokens += response2['total_prompt_tokens']
            whole_process_completion_tokens += response2['total_completion_tokens']
            whole_process_llm_costs += response2['total_cost_USD']
            whole_process_serper_credits += response2['total_serper_credits']
        else:
            valid_domains = set(response['valid_working_domains'])

        whole_process_prompt_tokens += response['total_prompt_tokens']
        whole_process_completion_tokens += response['total_completion_tokens']
        whole_process_llm_costs += response['total_cost_USD']
        whole_process_serper_credits += response['total_serper_credits']

        with open(log_file_paths['serper'], 'a') as f:
            f.write("\n\n")
            f.write(f"Validation domains.\n")
            f.write(f"Total Credits: {whole_process_serper_credits}\n")

        with open(log_file_paths['llm'], 'a') as f:
            f.write("\n\n")
            f.write(f"Validating domains:\n")
            f.write(f"Total Prompt Tokens: {whole_process_prompt_tokens}\n")
            f.write(f"Total Completion Tokens: {whole_process_completion_tokens}\n")
            f.write(f"Total Cost in USD: {whole_process_llm_costs}\n")

        st.write(f"Total Prompt Tokens: {whole_process_prompt_tokens}")
        st.write(f"Total Completion Tokens: {whole_process_completion_tokens}")
        st.write(f"Total Cost in USD: {whole_process_llm_costs}")
        st.write(f"Total Serper Credits: {whole_process_serper_credits}")

        endtime = datetime.now() - start_time

        common_values = set(gtd).intersection(valid_domains)
        common_values = list(common_values)
        
        missing_values_in_gtd = set(gtd).difference(valid_domains)
        missing_values_in_gtd = list(missing_values_in_gtd)
        
        new_values_in_valid_output = set(valid_domains).difference(gtd)
        new_values_in_valid_output = list(new_values_in_valid_output)
        
        accuracy = ((len(common_values) + len(new_values_in_valid_output)) / (len(gtd) + len(new_values_in_valid_output))) * 100

        max_length = max(len(agentsOutput), len(new_values_in_valid_output), len(gtd), len(common_values), len(valid_domains), len(missing_values_in_gtd))

        res_data = {
            "Company Name": [f"{company_name}"],
            'GTD': [len(gtd)],
            'AgentsOutput': [len(agentsOutput)],
            'Valid AgentsOutput': [len(list(valid_domains))],
            'Common Values': [len(common_values)],
            'Missing Values from GTD':[len(missing_values_in_gtd)],
            'New Values in Valid Output':[len(new_values_in_valid_output)],
            'Accuracy': [accuracy],
            'Folder': [result_directory]
        }

        res_df = pd.DataFrame(res_data)

        acc_df = pd.read_excel("validation/Accuracy_New.xlsx",engine = "openpyxl")
        acc_df = pd.concat([acc_df,res_df])
        acc_df.to_excel("validation/Accuracy_New.xlsx", index=False, engine="openpyxl")

        gtd = pad_list(gtd, max_length)
        agentsOutput = pad_list(agentsOutput, max_length)
        valid_agentsOutput = pad_list(list(valid_domains), max_length)
        common_values = pad_list(common_values, max_length)
        missing_values_in_gtd = pad_list(missing_values_in_gtd, max_length)
        new_values_in_valid_output = pad_list(new_values_in_valid_output, max_length)
        accuracy_pad = pad_list([accuracy], max_length)

        export_df = pd.DataFrame({
            'GTD': gtd,
            'AgentsOutput': agentsOutput,
            'Valid AgentsOutput': valid_agentsOutput,
            'Common Values': common_values,
            'Missing Values from GTD': missing_values_in_gtd,
            'New Values in Valid Output': new_values_in_valid_output,
            'Accuracy': accuracy_pad,
        })

        export_df.to_excel(os.path.join(final_results_directory, 'company_accuracy.xlsx'), index=False, header=True)

        with open(log_file_paths['log'], 'a') as f:
            f.write("\n\n")
            f.write(f"Time Taken To Complete the whole process: {endtime}\n")
            f.write(f"Total Prompt Tokens: {whole_process_prompt_tokens}\n")
            f.write(f"Total Completion Tokens: {whole_process_completion_tokens}\n")
            f.write(f"Total Cost in USD: {whole_process_llm_costs}\n")
            f.write(f"Total Serper Credits: {whole_process_serper_credits}\n")
            f.write(f"Accuracy: {accuracy}%\n")

        st.write(f"Time Taken To Complete the whole process: {endtime}")
        st.write(f"Accuracy: {accuracy}")