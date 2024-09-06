import streamlit as st
import requests
from dotenv import load_dotenv
import os
from company_structures import get_company_structures, get_company_structures_for_private_company
import multiprocessing
from functools import partial
from helpers import create_result_directory, extract_domain_name
from datetime import datetime
import pandas as pd
from company_structures_validation import validate_company_structure
from company_websites import get_official_websites, process_website_and_get_copyrights, process_copyright_research, process_domain_research, process_link_grabber

load_dotenv()

cost_per_1000_input_tokens = 0.005
cost_per_1000_output_tokens = 0.015

base_url = os.getenv('SEC_SEARCH_API_BASE_URL')
api_version = os.getenv('SEC_SEARCH_API_VERSION')

st.title("Market Research")

st.header("Company Search")

with st.form(key="company_search"):
    cik_number = st.text_input("CIK Number")
    company_name = st.text_input("Company Name")
    
    submit_button = st.form_submit_button(label="Submit")

if submit_button:
    try:
        response = requests.get(f"{base_url}{api_version}catalyst/sec/company?search={company_name}&cik_number={cik_number}&page=1&page_size=10")
        
        if response.status_code == 200:
            st.success("Companies fetched successfully!")
            st.write("API Response:", response.json())
        else:
            st.error(f"Failed to fetch companies form: {response.status_code}")
    except Exception as e:
        st.error(f"An error occurred: {e}")

st.header("Company Details Search")

with st.form(key="company_details_search"):
    sec_company_id = st.text_input("Company Id")
    company_name = st.text_input("Company Name")
    
    submit_button = st.form_submit_button(label="Submit")

if submit_button:
    try:
        start_time = datetime.now()

        folder_name = datetime.now().strftime("%Y%m%d%H%M%S")
        result_directory = f"{company_name}_{folder_name}"
        final_results_directory = f"final_results/{result_directory}"
        log_file_paths = create_result_directory(result_directory)

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

        try:
            response = requests.get(f"{base_url}{api_version}catalyst/sec/company/{sec_company_id}")
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            urls = []
        else:
            try:
                data = response.json()
                if 'data' in data:
                    st.success("Company details fetched successfully!")
                    urls = data['data']
                else:
                    st.error("Unexpected response format: 'data' key not found")
                    urls = []
            except ValueError as e:
                urls = []

        if len(urls) == 0:
            st.write("###### No url found for the given company")

            response = get_company_structures_for_private_company(company_name)

            print(response)
        else:
            st.write("###### Processing URLs for finding subsidiaries")
            progress_bar = st.progress(0)
            total_urls = len(urls)
            progress_step = 1 / total_urls

            fetch_company_structures_with_log = partial(get_company_structures, company_name, log_file_paths)

            with multiprocessing.Pool(processes=10) as pool:
                results = []
                for i, result in enumerate(pool.imap(fetch_company_structures_with_log, urls), 1):
                    results.append(result)
                    progress_bar.progress(i * progress_step)

            total_prompt_tokens = 0
            total_completion_tokens = 0
            total_cost_USD = 0.0

            company_structure_set = {company_name}

            for entry in results:
                if entry['result']['company_structure'] is not None:
                    company_structure_set.update(entry['result']['company_structure'])
                
                for exec_info in entry['exec_info']:
                    if exec_info['node_name'] == 'TOTAL RESULT':
                        total_prompt_tokens += exec_info.get('prompt_tokens', 0)
                        total_completion_tokens += exec_info.get('completion_tokens', 0)
                        total_cost_USD += exec_info.get('total_cost_USD', 0.0)

            with open(log_file_paths['llm'], 'a') as f:
                f.write(f"\n\n")
                f.write(f"Finding subsidiaries:\n")
                f.write(f"Total Prompt Tokens: {total_prompt_tokens}\n")
                f.write(f"Total Completion Tokens: {total_completion_tokens}\n")
                f.write(f"Total Cost in USD: {total_cost_USD}\n")

            st.write("###### Validating the subsidiaries")
            
            valid_subsidiaries = validate_company_structure(company_structure_set, company_name)

            st.markdown(f"<span style='color: green;'>####### {len(valid_subsidiaries['correct_agentsOutputList'])} valid subsidiaries found.</span>", unsafe_allow_html=True)

            input_cost = (valid_subsidiaries['llm_usage']['prompt_tokens'] / 1000) * cost_per_1000_input_tokens
            output_cost = (valid_subsidiaries['llm_usage']['completion_tokens'] / 1000) * cost_per_1000_output_tokens
            total_cost_USD = input_cost + output_cost

            with open(log_file_paths['llm'], 'a') as f:
                f.write(f"\n\n")
                f.write(f"Validating subsidiaries:\n")
                f.write(f"Total Prompt Tokens: {valid_subsidiaries['llm_usage']['prompt_tokens']}\n")
                f.write(f"Total Completion Tokens: {valid_subsidiaries['llm_usage']['completion_tokens']}\n")
                f.write(f"Total Cost in USD: {total_cost_USD}\n")

            export_df = pd.DataFrame({
                'AgentsOutput': valid_subsidiaries['agentsOutputList'],
                'ValidAgentsOutput': valid_subsidiaries['correct_agentsOutputList'],
                'ErrorAgentsOutput': valid_subsidiaries['error_agentsOutputList']
            })

            export_df['Accuracy'] = valid_subsidiaries['accuracy']
            export_df['Error'] = valid_subsidiaries['error']

            export_df.to_excel(os.path.join(final_results_directory, company_name.replace('/', '-') + '.xlsx'), index=False, header=True)

            st.write("###### Finding official websites for the subsidiaries")
            websites = get_official_websites(valid_subsidiaries['correct_agentsOutputList'], company_name, log_file_paths)

            input_cost = (websites['llm_usage']['prompt_tokens'] / 1000) * cost_per_1000_input_tokens
            output_cost = (websites['llm_usage']['completion_tokens'] / 1000) * cost_per_1000_output_tokens
            total_cost_USD = input_cost + output_cost

            with open(log_file_paths['llm'], 'a') as f:
                f.write("\n\n")
                f.write(f"Finding official websites for the subsidiaries:\n")
                f.write(f"Total Prompt Tokens: {websites['llm_usage']['prompt_tokens']}\n")
                f.write(f"Total Completion Tokens: {websites['llm_usage']['completion_tokens']}\n")
                f.write(f"Total Cost in USD: {total_cost_USD}\n")

            with open(log_file_paths['serper'], 'a') as f:
                f.write("\n\n")
                f.write(f"Finding official websites for the subsidiaries:\n")
                f.write(f"Total Credits: {websites['serper_credits']}\n")

            export_df = pd.DataFrame(websites['data'], columns=['Company Name', 'Website URL'])

            export_df.to_excel(os.path.join(final_results_directory, 'website_research_agent' + '.xlsx'), index=False, header=True)

            st.write("###### Find copyright for the subsidiaries")
            unique_urls = {entry['Website URL'] for entry in websites['data']}
            copyrights = process_website_and_get_copyrights(unique_urls, log_file_paths)

            file_path = os.path.join(final_results_directory, 'website_research_agent.xlsx')
            export_df = pd.read_excel(file_path)

            export_df['Copyright'] = export_df['Website URL'].map(copyrights['copyrights'])
            export_df.to_excel(file_path, index=False, header=True)

            st.write("###### Find websites using copyright")

            data = export_df[['Company Name', 'Copyright']]
            data_cleaned = data.dropna(subset=['Copyright'])
            unique_copyrights = data_cleaned.drop_duplicates(subset=['Copyright'])

            copyright_research = process_copyright_research(unique_copyrights, log_file_paths)
            with open(log_file_paths['serper'], 'a') as f:
                f.write("\n\n")
                f.write(f"Finding websites using copyright search:\n")
                f.write(f"Total Credits: {copyright_research['serper_credits']}\n")

            df = pd.DataFrame(copyright_research['copyright_results'], columns=['Website URL'])
            df.to_excel(os.path.join(final_results_directory, 'copyright_research_agent' + '.xlsx'), index=False, header=True)

            st.write("###### Find websites using domain search")
            domain_research = process_domain_research(unique_urls, log_file_paths)

            with open(log_file_paths['serper'], 'a') as f:
                f.write("\n\n")
                f.write(f"Finding websites using domain search:\n")
                f.write(f"Total Credits: {domain_research['serper_credits']}\n")

            df = pd.DataFrame(domain_research['domain_search_results'], columns=['Website URL'])
            df.to_excel(os.path.join(final_results_directory, 'domain_search_agent' + '.xlsx'), index=False, header=True)

            websites_results = set()

            for url in unique_urls:
                websites_results.add(extract_domain_name(url))

            combined_final_results = websites_results.union(copyright_research).union(domain_research)

            df = pd.DataFrame(combined_final_results, columns=['Website URL'])
            df.to_excel(os.path.join(final_results_directory, 'combined_final_results' + '.xlsx'), index=False, header=True)

            st.write("###### Start Link Grabber")

            link_grabber_results = process_link_grabber(unique_urls, log_file_paths)

            df = pd.DataFrame(link_grabber_results, columns=['Website URL'])
            df.to_excel(os.path.join(final_results_directory, 'link_grabber_agent' + '.xlsx'), index=False, header=True)

            endtime = datetime.now() - start_time

            with open(log_file_paths['log'], 'a') as f:
                f.write("\n\n")
                f.write(f"Time Taken To Complete the whole process: {endtime}\n")

            st.write(f"Time taken to run this whole process {endtime}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
