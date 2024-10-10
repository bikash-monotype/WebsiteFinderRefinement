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
from helpers import process_worker_function, extract_domain_name, is_working_domain, is_regional_domain_enhanced, translate_text, chunk_list, extract_main_part, social_media_domain_main_part, get_netloc, get_main_domain
from tools import search_multiple_page
import json_repair
from helpers import calculate_openai_costs, tokenize_text, is_reachable
import time
import pandas as pd
import json

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
            f.write(f"Exception when validating domain using scrapegraph AI for {domain}: {e}")
        print(f"Exception when validating domain using scrapegraph AI for {domain}: {e}")
        return {'domain': domain, 'isVisitable': 'No', 'reason': 'Exception when validating domain using scrapegraph AI', 'exec_info': None}
    
def validate_single_correct_domains(log_file_paths, main_company, main_company_domain, main_copyright_text, domain):
    try:
        if domain == extract_domain_name(main_company_domain):
            return {
                'results': [domain, 'Yes', 'Main company domain', '', ''],
                'llm_usage1': {
                    'prompt_tokens': 0,
                    'completion_tokens': 0
                },
                'llm_usage2': {
                    'prompt_tokens': 0,
                    'completion_tokens': 0,
                    'total_cost_USD': 0
                },
                'serper_credits': 0
            }

        total_serper_credits = 0

        total_prompt_tokens2 = 0
        total_completion_tokens2 = 0
        total_cost_USD2 = 0

        search_results = search_multiple_page(f"site:{domain} a part of {main_company}?", 10, 1, log_file_path=log_file_paths['log'])
        total_serper_credits += search_results['serper_credits']
        
        if len(search_results['all_results']) == 0:
            country_specific_domain = is_regional_domain_enhanced(domain)

            if country_specific_domain is True:
                translate_search_string = translate_text(f"site:{domain} a part of {main_company}?")

                if translate_search_string['is_translated'] == 'Yes':
                    search_results = search_multiple_page(translate_search_string['converted_text'], 10, 1, log_file_path=log_file_paths['log'])
                    total_serper_credits += search_results['serper_credits']

                    if len(search_results['all_results']) == 0:
                        return {
                            'results': [domain, 'No', 'No search results', '', ''],
                            'llm_usage1': {
                                'prompt_tokens': 0,
                                'completion_tokens': 0
                            },
                            'llm_usage2': {
                                'prompt_tokens': 0,
                                'completion_tokens': 0,
                                'total_cost_USD': 0
                            },
                            'serper_credits': total_serper_credits
                        }
                else:
                    return {
                        'results': [domain, 'No', 'No search results', '', ''],
                        'llm_usage1': {
                            'prompt_tokens': 0,
                            'completion_tokens': 0
                        },
                        'llm_usage2': {
                                'prompt_tokens': 0,
                                'completion_tokens': 0,
                                'total_cost_USD': 0
                            },
                        'serper_credits': total_serper_credits
                    }
            else:
                return {
                    'results': [domain, 'No', 'No search results', '', ''],
                    'llm_usage1': {
                        'prompt_tokens': 0,
                        'completion_tokens': 0
                    },
                    'llm_usage2': {
                        'prompt_tokens': 0,
                        'completion_tokens': 0,
                        'total_cost_USD': 0
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

                If the relationship is valid, return 'Yes'. If not, return 'No' and provide appropriate reason for your decision.

                Only use information from the search results. Avoid assumptions.

                Output format:
                ['{domain}', 'Yes/No', 'Reason']

                **Scoring:**
                - +1 for correct output (based on evidence)
                - -1 for incorrect or speculative output
                """
            ),
            agent=domain_company_validation_researcher,
            expected_output="['{domain}', 'Yes/No', 'Reason']"
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

                If the relationship is valid, return 'Yes'. If not, return 'No' and provide appropriate reason for your decision.

                Only use information from the search results. Avoid assumptions.

                Output format:
                ['{domain}', 'Yes/No', 'Reason']

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
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens
        }

        results = json_repair.loads(results.raw)

        with open(log_file_paths['crew_ai'], 'a') as f:
            f.write("\n")
            f.write(f"{domain} validation" + str(llm_usage))

        search_results_links = [get_netloc(result['link']) for result in search_results['all_results']]

        if results[1] == 'Yes':
            final_validation = {}

            if domain not in search_results_links:
                final_validation['is_company_domain'] = 'No'
                final_validation['reason'] = 'Domain not found in search results but only subdomain found.'
                final_validation['ownership_not_clear'] = 'Yes'
                final_validation['link'] = ''
            else:
                main_url = get_main_domain(search_results['all_results'][0]['link'])

                final_validation = validate_domains_that_are_considered_correct_by_llm_in_google_search(main_url, main_company, main_company_domain, main_copyright_text, log_file_paths)

                if final_validation['graph_exec_info'] is not None:
                    for exec_info in final_validation['graph_exec_info']:
                        if exec_info['node_name'] == 'TOTAL RESULT':
                            total_prompt_tokens2 = exec_info.get('prompt_tokens', 0)
                            total_completion_tokens2 = exec_info.get('completion_tokens', 0)
                            total_cost_USD2 = exec_info.get('total_cost_USD', 0.0)

                    if main_url != search_results['all_results'][0]['link'].rstrip('/'):
                        if final_validation['ownership_not_clear'] == 'Yes':
                            final_validation = validate_domains_that_are_considered_correct_by_llm_in_google_search(search_results['all_results'][0]['link'], main_company, main_company_domain, main_copyright_text, log_file_paths)

                            if final_validation['graph_exec_info'] is not None:
                                for exec_info in final_validation['graph_exec_info']:
                                    if exec_info['node_name'] == 'TOTAL RESULT':
                                        total_prompt_tokens2 += exec_info.get('prompt_tokens', 0)
                                        total_completion_tokens2 += exec_info.get('completion_tokens', 0)
                                        total_cost_USD2 += exec_info.get('total_cost_USD', 0.0)
                    else:
                        if ((len(search_results['all_results']) > 1) and (final_validation['ownership_not_clear'] == 'Yes')):
                            final_validation = validate_domains_that_are_considered_correct_by_llm_in_google_search(search_results['all_results'][1]['link'], main_company, main_company_domain, main_copyright_text, log_file_paths)

                            if final_validation['graph_exec_info'] is not None:
                                for exec_info in final_validation['graph_exec_info']:
                                    if exec_info['node_name'] == 'TOTAL RESULT':
                                        total_prompt_tokens2 += exec_info.get('prompt_tokens', 0)
                                        total_completion_tokens2 += exec_info.get('completion_tokens', 0)
                                        total_cost_USD2 += exec_info.get('total_cost_USD', 0.0)

            return {
                'results': [domain, final_validation['is_company_domain'], final_validation['reason'], final_validation['ownership_not_clear'], final_validation['link']],
                'llm_usage1': {
                    'prompt_tokens': llm_usage['prompt_tokens'],
                    'completion_tokens': llm_usage['completion_tokens']
                },
                'llm_usage2': {
                    'prompt_tokens': total_prompt_tokens2,
                    'completion_tokens': total_completion_tokens2,
                    'total_cost_USD': total_cost_USD2
                },
                'serper_credits': total_serper_credits
            }

        return {
            'results': [results[0], results[1], results[2], '', ''],
            'llm_usage1': {
                'prompt_tokens': llm_usage['prompt_tokens'],
                'completion_tokens': llm_usage['completion_tokens']
            },
            'llm_usage2': {
                'prompt_tokens': total_prompt_tokens2,
                'completion_tokens': total_completion_tokens2,
                'total_cost_USD': total_cost_USD2
            },
            'serper_credits': total_serper_credits
        }
    except Exception as e:
        with open(log_file_paths['log'], 'a') as f:
            f.write(f"Exception when validating domain using crew AI: {e}")
        return {'results': [domain, 'No', f'Exception when validating domain using crew AI: {e}', '', ''], 'llm_usage1': {'prompt_tokens': 0, 'completion_tokens': 0}, 'llm_usage2': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_cost_USD': 0}, 'serper_credits': 0}

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

def validate_domains_that_are_considered_correct_by_llm_in_google_search(url, main_company, main_company_domain, main_copyright_text, log_file_paths):
    try :
        sample_json_output = json.dumps({'is_company_domain': 'Yes/No', 'ownership_not_clear': 'Yes/No', 'reason': 'Reason for Yes or No value in is_company_domain'})

        prompt = (
            f"""
            **Task Objective:**
            Determine whether the specified domain is **formally associated** with the company owning the main website by being part of its subsidiaries, brands, or formal partnerships involving **ownership stakes or equity relationships**.
            Exclude any instances where the domain appears as a third-party tool, service, advertisement, or reporting platform, or is associated through license agreements or non-equity partnerships, without formal ownership or stake-based association.

            **Domain:** {url}  
            **Main Company:** {main_company}  
            **Main Company Domain:** {main_company_domain}

            ### **Task Breakdown:**

            #### 1. **Presence Verification:**
            - **Search the provided website ({url})** to determine if it is part of or associated with the specified company ({main_company}).

            #### 2. **Ownership and Association Analysis of {main_company}:**

            ##### **Strict Copyright Matching:**
            - Start by checking the copyright section, usually in the website's footer, for explicit ownership details.
            - **If {main_copyright_text} is not None, first try to find an exact match of "© {main_copyright_text}" without considering the year. If an exact match is not found, then check if the copyright text includes a hyperlink to the main company's domain ({main_company_domain}) and mentions the company explicitly.** The hyperlink for phrases similar to "Part of {main_company}" (e.g., "A member of {main_company}", "Owned by {main_company}", or any similar phrase indicating formal ownership) must point to the main company's domain ({main_company_domain}) to confirm the association.
            - **The main company name must be exact; nothing else is allowed.** For example, "abc inc" and "abc xyz inc" should be treated differently and are not considered exact matches.

            ##### **Focus on Textual Ownership Mentions:**
            - Look for textual confirmation of ownership, especially in legal documents or the "About Us" section, but only after attempting the copyright match.

            ##### **Confirm if the domain is associated with the company in one of the following ways:**
            - **Subsidiaries:** Verify if the domain belongs to a subsidiary of the company. Look for corporate ownership listings or related documentation.
            - **Brands:** Identify if the domain represents a brand of the company. This includes domains for products, services, or divisions under the company’s umbrella.
            - **Partnerships with Stakes:** Determine if the domain is part of a formal partnership where the company holds stakes or equity-based investments.
            - **Acquisitions:** Check if the domain was acquired or merged into the company.

            #### 3. **Cross-Verification:**
            - **Verify Ownership Consistency:** Ensure that all sections of the website (e.g., About Us, Contact, Footer) consistently indicate ownership by {main_company} or its subsidiaries/brands.
            - **Check Domain Registration:** Where possible, verify the domain registration details to confirm ownership by the company or its affiliates.

            #### 4. **Handling Third-Party Services and Tools:**
            - **Exclude third-party tools or services** (e.g., analytics, marketing, IT services) unless they are owned or directly managed by the company or its subsidiaries.
            - If the domain belongs to a **third-party tool or service** without ownership ties (e.g., external financial platforms, marketplaces, or unrelated service providers), it should be excluded as not formally associated.

            #### 5. **Exclusion of Non-Ownership Mentions:**
            - **Exclude any domain that appears as a**:
            - **Service provider**
            - **Advertisement platform**
            - **Reporting platform**
            - **Tool**
            - **Without ownership, branding, or equity-based partnerships with the company.**

            #### 6. **Verification of Explicit Ownership Indicators:**
            - **Primary Focus on Copyright Match:**
            - **Footer:** First, try to find an exact match of "{main_copyright_text}" in the footer. If an exact match is not found, check if the copyright text includes a hyperlink to the main company's domain ({main_company_domain}) and explicitly mentions the company with phrases similar to "Part of [main_company]" (e.g., "A member of [main_company]", "Owned by [main_company]", or any similar phrase indicating formal ownership). The hyperlink for the phrase "Part of [main_company]" must point to the main company's domain ({main_company_domain}) to confirm the association.
            - **About Us & Legal Notices:** Only consider these sections if the copyright match is found or if the main company's domain is included and the company is explicitly mentioned in the footer.
            - **Avoid Relying Solely on Visual Elements:**  
            Do not assume ownership based solely on visual branding or logos without textual confirmation.

            #### 7. **Verification of Ownership Clarity:**
            - If **explicit ownership details are not available** or the relationship is unclear, set the `ownership_not_clear` field to "Yes."
            - If ownership is clearly stated or verified through the copyright match or mention of the main company's domain, set `ownership_not_clear` to "No."

            ### **Output Format:**
            Return the following JSON format: {sample_json_output}

            ### **Important Instructions:**

            #### **Strict Copyright Matching**
            - First, try to find an exact match of "{main_copyright_text}" in the footer, ignoring the year. If an exact match is not found, check if the copyright text includes a hyperlink to the main company's domain ({main_company_domain}) and explicitly mentions the company. The hyperlink for the phrase "Part of [main_company]" must point to the main company's domain ({main_company_domain}).
            - Do not accept variations like additional text or company suffixes (e.g., "Inc.", "LLC").
            - **The main company name must be exact; nothing else is allowed.** For example, "abc inc" and "abc xyz inc" should be treated differently and are not considered exact matches.

            #### **Prioritize Ownership and Association Analysis:**
            - After attempting the copyright match, proceed to identify explicit ownership, brand affiliation, subsidiaries, or equity-based partnerships.

            #### **Exclude License-Based Associations:**
            - **Domains associated through licensing agreements without ownership or equity stakes should be excluded.** This includes situations where the company provides services under license (e.g., financial cards, prepaid cards, bank accounts) but does not explicitly own or control the domain. Such products or services offered under a license agreement without explicit ownership must always be excluded and classified as **No** for formal ownership. Mentions like 'issued by', 'established by', or 'provided by' are never valid and should directly lead to setting `ownership_not_clear` to **Yes**. Only explicit language indicating ownership, such as 'owned by' or 'a subsidiary of', is valid. Additionally, check the copyright section for explicit ownership details by trying to find an exact match of "© {main_copyright_text}" in the footer, ignoring the year, as copyright information can often clarify ownership. Any mention of the domain being 'issued by,' 'established by,' or under license must always be classified as **No** for formal ownership, regardless of context. These mentions do not indicate formal ownership or control and must be explicitly marked as insufficient.

            #### **Exclude Reporting Platforms and Unrelated Tools/Services:**
            - Websites that report on or discuss the company's activities or offer unrelated services without formal ownership should be excluded.

            #### **Focus on Ownership and Stakes:**
            - Only include domains that are tied to the company through ownership stakes or equity-based relationships. Exclude collaborations, licenses, or third-party agreements without ownership.

            #### **Avoid Assumptions:**
            - Do not infer associations based on partial information or general references. Only classify as "Yes" when there is clear evidence of formal ownership or equity-based association.

            #### **Ownership Clarity:**
            - Set `ownership_not_clear` to "Yes" if the ownership is not clearly mentioned; otherwise, set it to "No."

            **YOU CANNOT MAKE ANY ASSUMPTIONS. Exclude the domain if ownership is tied to an individual rather than the company entity.**
            """
        )

        smart_scraper_graph = SmartScraperGraph(
            prompt=prompt,
            source=url,
            config=graph_config,
        )

        result = smart_scraper_graph.run()
        graph_exec_info = smart_scraper_graph.get_execution_info()

        return {
            'is_company_domain': result['is_company_domain'],
            'ownership_not_clear': result['ownership_not_clear'],
            'reason': result['reason'],
            'link': url,
            'graph_exec_info': graph_exec_info
        }
    except Exception as e:
        with open(log_file_paths['log'], 'a') as f:
            f.write(f"Exception when validating domain {url} using scrapegraph AI: {e}")
        print(f"Exception when validating domain {url} using scrapegraph AI: {e}")
        return {
            'is_company_domain': 'No',
            'ownership_not_clear': 'Yes',
            'reason': f'Exception when validating domain {url} using scrapegraph AI',
            'link': url,
            'graph_exec_info': None
        }

def validate_single_correct_linkgrabber_domains(log_file_paths, main_company, company_domain, domain_key_value):
    main_domain, domain = domain_key_value

    if extract_domain_name(company_domain) == domain:
        return {
            'main_domain': main_domain,
            'domain': domain,
            'link': '',
            'valid': 'Yes',
            'reason': 'Main domain',
            'graph_exec_info': None,
            'total_serper_credits': 0
        }

    total_serper_credits = 0

    search_results = search_multiple_page(f"site:{main_domain} {domain}", 10, 1, log_file_path=log_file_paths['log'])
    total_serper_credits += search_results['serper_credits']

    if len(search_results['all_results']) == 0:
        return {
            'main_domain': main_domain,
            'domain': domain,
            'link': '',
            'valid': 'No',
            'reason': 'Zero search results',
            'graph_exec_info': None,
            'total_serper_credits': total_serper_credits
        }
    
    try :
        sample_json_output = json.dumps({'valid': 'Yes/No', 'reason': 'Reason for Yes or No value'})

        prompt = (
            f"""
                **Task Objective:**  
                Determine whether the specified domain is **formally and currently associated** with the company owning the main website by being part of its subsidiaries, brands, or formal partnerships involving **ownership stakes or equity relationships**. **Exclude any instances where the domain appears as a third-party tool, service, advertisement, or reporting platform, or is associated through license agreements or non-equity partnerships, without formal ownership or stake-based association.**

                ### **Task Breakdown:**

                1. **Presence Verification:**
                - Search the provided website (`{main_domain}`) to check if the specified domain (`{domain}`) appears anywhere on the site.
                - **Email Address Cross-Verification:** If the email address of the domain (`{domain}`) appears in contact information (e.g., contact@{domain}), prioritize this as a potential indicator of ownership or formal association with the company.

                2. **Ownership and Association Analysis of {main_company}:**
                - **Confirm if the domain is associated with the company** in one of the following ways:
                    - **Subsidiaries:** Verify if the domain belongs to a subsidiary of the company. Look for corporate ownership listings or related documentation.
                    - **Brands:** Identify if the domain represents a brand of the company. This includes domains for products, services, or divisions under the company’s umbrella.
                    - **Partnerships with Stakes:** Determine if the domain is part of a formal partnership where the company holds stakes or equity-based investments.
                    - **Acquisitions:** Check if the domain was acquired or merged into the company.
                - **Ownership Indicators:** Look for explicit mentions of ownership, such as:
                    - "operated by"
                    - "owned by"
                    - "subsidiary of"
                    - "brand of"
                    - "division of"
                    - "partnership with equity stake"
                - **Cross-Verification:**
                    - Contact 

                - **Important Exclusion - Financial Services Offered Under License Agreements:**
                    - **Exclude domains associated through license agreements** or where the company provides financial services (e.g., deposit accounts, debit card issuance) under a license to another entity. These do not constitute formal ownership or equity relationships.
                    - **Examples to Exclude:**
                    - If the company is merely providing banking services for accounts or cards issued by another entity under a licensing agreement.
                    - If the domain's services are established or issued by the company under a license but are not owned or controlled by the company.
                    
                4. **Handling Third-Party Services and Tools:**
                - **Exclude third-party tools or services** (e.g., analytics, marketing, IT services) unless they are owned or directly managed by the company or its subsidiaries.
                - If the domain belongs to a **third-party tool or service** without ownership ties (e.g., external financial platforms, marketplaces, or unrelated service providers), it should be excluded as not formally associated.

                5. **Exclusion of Non-Ownership Mentions:**
                - **Exclude any domain that appears as a**:
                    - **Service provider**
                    - **Advertisement platform**
                    - **Reporting platform**
                    - **Tool**
                    - **Without ownership, branding, or equity-based partnerships with the company.**
                - **Exclude Mentions Under License Agreements:**
                    - **Mentions such as "established by", "issued by", or "in partnership with" under a license agreement do not indicate formal ownership.** Unless the company has a direct ownership stake or the domain is a subsidiary or brand, it should be excluded.
                    - **Exclude domains where the company acts under a license agreement without ownership or equity stakes.**

                ### **Output Format:**
                Return the following JSON format:
                {sample_json_output}

                ### **Important Instructions:**
                - **Prioritize Ownership and Association Analysis:**  
                Focus primarily on identifying explicit ownership, brand affiliation, subsidiaries, or equity-based partnerships. Do not rely on general mentions, licensing agreements, or references.
                - **Exclude License-Based Associations:**  
                **Domains associated through licensing agreements without ownership or equity stakes should be excluded.** This includes situations where the company provides services under license but does not own the domain.
                - **Exclude Reporting Platforms:**  
                Websites that report on or discuss the company’s activities but are not owned or operated by the company should be excluded.
                - **Exclude Unrelated Tools/Services:**  
                If the website is operated by a third party without ownership or stakes (e.g., unrelated financial platforms or services), exclude it from being formally associated with the company.
                - **Focus on Ownership and Stakes:**  
                Only include domains that are tied to the company through ownership stakes or equity-based relationships. Exclude collaborations, licenses, or third-party agreements without ownership.
                - **Avoid Assumptions:**  
                Do not infer associations based on partial information or general references. Only classify as "Yes" when there is clear evidence of formal ownership or equity-based association.

                YOU CANNOT MAKE ANY ASSUMPTIONS.
            """
        )
        
        smart_scraper_graph = SmartScraperGraph(
            prompt=prompt,
            source=search_results['all_results'][0]['link'],
            config=graph_config,
        )

        result = smart_scraper_graph.run()
        graph_exec_info = smart_scraper_graph.get_execution_info()

        if result['valid'] == 'Yes':
            is_reachable_domain = is_reachable(domain)

            if is_reachable_domain is False:
                return {
                    'main_domain': main_domain,
                    'domain': domain,
                    'reason': 'Domain is valid but not not reachable',
                    'link': search_results['all_results'][0]['link'],
                    'valid': 'No',
                    'graph_exec_info': graph_exec_info,
                    'total_serper_credits': total_serper_credits
                }

        return {
            'main_domain': main_domain,
            'domain': domain,
            'reason': result['reason'],
            'link': search_results['all_results'][0]['link'],
            'valid': result['valid'],
            'graph_exec_info': graph_exec_info,
            'total_serper_credits': total_serper_credits
        }
    except Exception as e:
        with open(log_file_paths['log'], 'a') as f:
            f.write(f"Exception when validating {domain} using scrapegraph AI: {e}")
        print(f"Exception when validating {domain} using scrapegraph AI: {e}")
        return {
            'main_domain': main_domain,
            'domain': domain,
            'link': search_results['all_results'][0]['link'],
            'valid': 'No',
            'reason': f'Exception when validating domain using scrapegraph AI: {e}',
            'graph_exec_info': None,
            'total_serper_credits': total_serper_credits
        }

def validate_linkgrabber_domains(main_company, company_domain, domains, log_file_path):
    domains_key_value = []

    st.write('###### Remove incorrect linkgrabber domains')

    for main_domain, value in domains.items():
        for domain in value:
            if domain != value and not pd.isna(domain) and isinstance(domain, str) and domain != "." and extract_main_part(domain) not in social_media_domain_main_part:
                domains_key_value.append((main_domain, domain))

    progress_bar = st.progress(0)
    total_domains = len(domains_key_value)
    progress_step = 1 / total_domains

    validate_single_correct_domains_with_log = partial(validate_single_correct_linkgrabber_domains, log_file_path, main_company, company_domain)

    serialized_function = dill.dumps(validate_single_correct_domains_with_log)

    chunk_size = 15
    results = []

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_cost_USD = 0.0
    total_serper_credits = 0

    valid_working_domains = set()
    invalid_non_working_domains = set()

    with multiprocessing.Pool(processes=20) as pool:
        for chunk in chunk_list(domains_key_value, chunk_size):
            chunk_results = pool.map(partial(process_worker_function, serialized_function), chunk)
            results.extend(chunk_results)
            progress_bar.progress(min((len(results) + 1) * progress_step, 1.0))

    validation_domain_with_reason = []

    for res in results:
        validation_domain_with_reason.append([res['main_domain'], res['domain'], res['valid'], res['reason'], res['link']])

        if res['valid'] == 'Yes':
            valid_working_domains.add(res['domain'])
        else:
            invalid_non_working_domains.add(res['domain'])

        total_serper_credits += res['total_serper_credits']

        if res['graph_exec_info'] is not None:
            for exec_info in res['graph_exec_info']:
                if exec_info['node_name'] == 'TOTAL RESULT':
                    total_prompt_tokens += exec_info.get('prompt_tokens', 0)
                    total_completion_tokens += exec_info.get('completion_tokens', 0)
                    total_cost_USD += exec_info.get('total_cost_USD', 0.0)

    with open(log_file_path['llm'], 'a') as f:
        f.write('Validate linkgrabber domains')
        f.write(f"Total prompt tokens: {total_prompt_tokens}\n")
        f.write(f"Total completion tokens: {total_completion_tokens}\n")
        f.write(f"Total cost: {total_cost_USD}\n")

    with open(log_file_path['serper'], 'a') as f:
        f.write('Validate linkgrabber domains')
        f.write(f"Total Credits: {total_serper_credits}\n")

    return {
        'link_grabber_validation_AI_responses': validation_domain_with_reason,
        'valid_working_domains': list(valid_working_domains),
        'invalid_non_working_domains': list(invalid_non_working_domains),
        'total_prompt_tokens': total_prompt_tokens,
        'total_completion_tokens': total_completion_tokens,
        'total_cost_USD': total_cost_USD,
        'total_serper_credits': total_serper_credits
    }

def validate_agentsOutput_domains(domains, main_company, main_company_domain, main_copyright_text, log_file_path):
    domains = [value for value in domains if not pd.isna(value) and isinstance(value, str) and value != "." and extract_main_part(value) not in social_media_domain_main_part]

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_cost_USD = 0.0

    total_prompt_tokens2 = 0
    total_completion_tokens2 = 0
    total_serper_credits = 0
    total_cost_USD2 = 0

    st.write('###### Remove incorrect agentsOutput domains')

    progress_bar = st.progress(0)
    total_domains = len(domains)
    progress_step = 1 / total_domains

    validate_single_correct_domains_with_log = partial(validate_single_correct_domains, log_file_path, main_company, main_company_domain, main_copyright_text)

    serialized_function = dill.dumps(validate_single_correct_domains_with_log)

    valid_working_domains = set()
    invalid_non_working_domains = set()

    chunk_size = 15
    results = []

    with multiprocessing.Pool(processes=20) as pool:
        for chunk in chunk_list(domains, chunk_size):
            chunk_results = pool.map(partial(process_worker_function, serialized_function), chunk)
            results.extend(chunk_results)
            progress_bar.progress(min((len(results) + 1) * progress_step, 1.0))

    validation_domain_with_reason = []

    for res in results:
        if isinstance(res['results'], list) and len(res['results']) >= 5:
            if res['results'][1] != 'Yes':
                invalid_non_working_domains.add(res['results'][0])
            else:
                valid_working_domains.add(res['results'][0])

            validation_domain_with_reason.append([res['results'][0], res['results'][3], res['results'][1], res['results'][4], res['results'][2]])
        else:
            with open(log_file_path['log'], 'a') as f:
                f.write(f"Unexpected format in results: {res['results']}")

        total_prompt_tokens += res['llm_usage1']['prompt_tokens']
        total_completion_tokens += res['llm_usage1']['completion_tokens']
        total_serper_credits += res['serper_credits']

        total_prompt_tokens2 += res['llm_usage2']['prompt_tokens']
        total_completion_tokens2 += res['llm_usage2']['completion_tokens']
        total_cost_USD2 += res['llm_usage2']['total_cost_USD']

    total_cost_USD += calculate_openai_costs(total_prompt_tokens, total_completion_tokens)

    with open(log_file_path['llm'], 'a') as f:
        f.write('Remove incorrect domains')
        f.write(f"Total prompt tokens: {total_prompt_tokens}\n")
        f.write(f"Total completion tokens: {total_completion_tokens}\n")
        f.write(f"Total cost: {total_cost_USD}\n")

    with open(log_file_path['llm'], 'a') as f:
        f.write('Remove incorrect domains (2nd validation)')
        f.write(f"Total prompt tokens: {total_prompt_tokens2}\n")
        f.write(f"Total completion tokens: {total_completion_tokens2}\n")
        f.write(f"Total cost: {total_cost_USD2}\n")

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
        'agentsOutput_validation_AI_responses': validation_domain_with_reason,
        'valid_working_domains': list(valid_working_domains),
        'invalid_non_working_domains': list(invalid_non_working_domains),
        'total_prompt_tokens': total_prompt_tokens + total_prompt_tokens2,
        'total_completion_tokens': total_completion_tokens + total_completion_tokens2,
        'total_cost_USD': total_cost_USD + total_cost_USD2,
        'total_serper_credits': total_serper_credits
    }
