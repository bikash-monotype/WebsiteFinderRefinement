import streamlit as st
import pandas as pd
from datetime import datetime
from helpers import create_result_directory, extract_domain_name, pad_list, social_media_domain_main_part, extract_main_part
import os
from company_websites_validation import validate_domains, validate_working_domains

st.title("Accuracy check without GTD")

uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])

company_name = st.text_input("Enter company name")

if uploaded_file is not None and company_name is not None:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    if st.button("Process File"):
        print("Processing file...")
        start_time = datetime.now()
        # gtd = df['GTD'].tolist()
        agentsOutput = df['AgentsOutput'].tolist()

        # gtd = [value for value in gtd if not pd.isna(value)]
        agentsOutput = [value for value in agentsOutput if not pd.isna(value) and isinstance(value, str) and value != "." and extract_main_part(value) not in social_media_domain_main_part]

        # gtd = [extract_domain_name(url) for url in gtd]
        # gtd = set(gtd)
        # gtd = list(gtd)

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

        response = validate_domains(agentsOutput, company_name, log_file_paths)

        st.write('###### Validate Ground Truth')

        # response2 = validate_working_domains(gtd, log_file_paths)

        # valid_gtd = response2['valid_working_domains']

        whole_process_prompt_tokens += response['total_prompt_tokens']
        whole_process_completion_tokens += response['total_completion_tokens']
        whole_process_llm_costs += response['total_cost_USD']
        whole_process_serper_credits += response['total_serper_credits']

        # whole_process_prompt_tokens += response2['total_prompt_tokens']
        # whole_process_completion_tokens += response2['total_completion_tokens']
        # whole_process_llm_costs += response2['total_cost_USD']

        # export_df = pd.DataFrame({
        #     'Domains': response['invalid_non_working_domains'].keys(),
        #     'Reason': response['invalid_non_working_domains'].values()
        # })

        # export_df.to_excel(os.path.join(final_results_directory, 'invalid_non_working_domains.xlsx'), index=False, header=True)

        # export_df = pd.DataFrame({
        #     'Domains': response2['invalid_non_working_domains'].keys(),
        #     'Reason': response2['invalid_non_working_domains'].values()
        # })

        # export_df.to_excel(os.path.join(final_results_directory, 'invalid_gtd.xlsx'), index=False, header=True)

        export_df = pd.DataFrame({
            'Domains': response['final_invalid_non_working_domains'],
        })

        export_df.to_excel(os.path.join(final_results_directory, 'final_invalid_non_working_domains.xlsx'), index=False, header=True)

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

        # common_values = set(gtd).intersection(response['final_valid_working_domains'])
        # common_values = list(common_values)
        #
        # missing_values_in_gtd = set(gtd).difference(response['final_valid_working_domains'])
        # missing_values_in_gtd = list(missing_values_in_gtd)
        #
        # new_values_in_valid_output = set(response['final_valid_working_domains']).difference(gtd)
        # new_values_in_valid_output = list(new_values_in_valid_output)
        #
        # accuracy = ((len(common_values) + len(new_values_in_valid_output)) / (len(gtd) + len(new_values_in_valid_output))) * 100

        # max_length = max(len(gtd), len(agentsOutput), len(new_values_in_valid_output), len(gtd), len(common_values))
        max_length = max(len(agentsOutput))

        res_data = {
            "Company Name": [f"{company_name}"],
            # 'GTD': [len(gtd)],
            # 'Valid GTD': [len(valid_gtd)],
            # "Invalid GTD":[len(list(response2['invalid_non_working_domains'].keys()))],
            'AgentsOutput': [len(agentsOutput)],
            'Valid AgentsOutput': [len(list(response['final_valid_working_domains']))],
            # 'Common Values': [len(common_values)],
            # 'Missing Values from GTD':[len(missing_values_in_gtd)],
            # 'New Values in Valid Output':[len(new_values_in_valid_output)],
            # 'Accuracy': [accuracy],
            'Folder': [result_directory]
        }

        res_df = pd.DataFrame(res_data)

        acc_df = pd.read_excel("validation/response.xlsx",engine = "openpyxl")
        acc_df = pd.concat([acc_df,res_df])
        acc_df.to_excel("validation/response.xlsx", index=False, engine="openpyxl")

        # gtd = pad_list(gtd, max_length)
        # valid_gtd = pad_list(valid_gtd, max_length)
        # invalid_gtd = pad_list(list(response2['invalid_non_working_domains'].keys()), max_length)
        agentsOutput = pad_list(agentsOutput, max_length)
        valid_agentsOutput = pad_list(list(response['final_valid_working_domains']), max_length)
        # common_values = pad_list(common_values, max_length)
        # missing_values_in_gtd = pad_list(missing_values_in_gtd, max_length)
        # new_values_in_valid_output = pad_list(new_values_in_valid_output, max_length)
        # accuracy_pad = pad_list([accuracy], max_length)

        export_df = pd.DataFrame({
            # 'GTD': gtd,
            # 'Valid GTD': valid_gtd,
            # 'Invalid GTD': invalid_gtd,
            'AgentsOutput': agentsOutput,
            'Valid AgentsOutput': valid_agentsOutput,
            # 'Common Values': common_values,
            # 'Missing Values from GTD': missing_values_in_gtd,
            # 'New Values in Valid Output': new_values_in_valid_output,
            # 'Accuracy': accuracy_pad,
        })

        export_df.to_excel(os.path.join(final_results_directory, 'company_accuracy.xlsx'), index=False, header=True)

        with open(log_file_paths['log'], 'a') as f:
            f.write("\n\n")
            f.write(f"Time Taken To Complete the whole process: {endtime}\n")
            f.write(f"Total Prompt Tokens: {whole_process_prompt_tokens}\n")
            f.write(f"Total Completion Tokens: {whole_process_completion_tokens}\n")
            f.write(f"Total Cost in USD: {whole_process_llm_costs}\n")
            f.write(f"Total Serper Credits: {whole_process_serper_credits}\n")
            # f.write(f"Accuracy: {accuracy}%\n")

        st.write(f"Time Taken To Complete the whole process: {endtime}")
        # st.write(f"Accuracy: {accuracy}")