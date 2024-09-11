from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
import time
import validators
import re
import os
import urllib.parse
import tldextract
import dill
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

social_media_domains = [
    'facebook.com', 'twitter.com', 'instagram.com', 'cookiepedia.co.uk', 'fonts.googleapis.com', 'jwt.io', 'google-analytics.com', 'adobe.com',
    'threads.net', 'linkedin.com', 'pinterest.com', 'youtube.com', 'onetrust.com', 'amazon.com', 'reddit.com', 'wordpress.org', 'adobe.io',
    'tiktok.com', 'snapchat.com', 'whatsapp.com', 'quora.com', 'google.com', 'github.com', 'apple.com', 'vimeo.com', 'youtu.be', 'cloudflare.net', 'goo.gl', 'mozilla.org', 'maps.app.goo.gl'
]

def process_worker_function(serialized_func, row):
    func = dill.loads(serialized_func)
    return func(row)

def extract_main_part(url):
    parsed_url = urllib.parse.urlparse(url)
    extracted = tldextract.extract(parsed_url.netloc)
    domain_name = extracted.domain

    return domain_name

def calculate_openai_costs(input_tokens, output_tokens):
    input_cost = (input_tokens / 1000) * float(os.getenv('AZURE_OPENAI_MODEL_INPUT_TOKENS_COST'))
    output_cost = (output_tokens / 1000) * float(os.getenv('AZURE_OPENAI_MODEL_OUTPUT_TOKENS_COST'))
    return (input_cost + output_cost)

def get_main_domain(url):
    parsed_url = urllib.parse.urlparse(url)
    return f"{parsed_url.scheme}://{parsed_url.netloc}"

def is_social_media_link(link):
    return any(domain in link for domain in social_media_domains)

def create_log_file(file_path):
    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            f.write('')

def remove_trailing_slash(links):
    return [link.rstrip('/') for link in links]

def pad_list(lst, length):
    return lst + [None] * (length - len(lst))

def get_scrapegraph_config():
    azure_model = AzureChatOpenAI(
        openai_api_version=os.getenv('OPENAI_API_VERSION'),
        azure_deployment=os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME'),
        azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
        api_key=os.getenv('AZURE_OPENAI_API_KEY'),
        temperature=0
    )

    azure_embeddings = AzureOpenAIEmbeddings(
        azure_deployment=os.getenv('AZURE_OPENAI_EMBEDDINGS'),
        openai_api_version="2023-05-15",
        azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
        api_key=os.getenv('AZURE_OPENAI_API_KEY'),
    )

    return {
        "llm": {
            "model_instance": azure_model,
            "model_tokens": 100000,
        },
        "embeddings": {
            "model_instance": azure_embeddings
        },
        "verbose": True,
        "headless": False,
    }

# this function will be used to detect redirection and not reachable domains.
def is_working_domain(url, log_file_paths):
    max_retries = 3
    valid_domain = True
    reason = ''

    for attempt in range(max_retries):
        with sync_playwright() as p:
            try:
                browser = p.chromium.launch(headless=False, args=['--disable-http2'])
                page = browser.new_page()
                page.goto(url, timeout=120000)
                
                page.wait_for_load_state('load')
                time.sleep(10)

                if extract_domain_name(url) != extract_domain_name(page.url):
                    valid_domain = False
                    reason = 'Redirection.'
                break
            except PlaywrightTimeoutError as te:
                with open(log_file_paths['log'], 'a') as f:
                    f.write(f"Timeout error processing {url} on attempt {attempt + 1}: {te}")
                print(f"Timeout error processing {url} on attempt {attempt + 1}: {te}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    valid_domain = False
                    reason = f"Timeout error processing url {url}"
            except Exception as e:
                with open(log_file_paths['log'], 'a') as f:
                    f.write(f"Error processing {url}: {e}")
                print(f"Error processing {url}: {e}")
                valid_domain = False
                reason = f"Error processing {url}: {e}"
            finally:
                if page is not None:
                    page.close()
    return {
        'is_valid': valid_domain,
        'reason': reason
    }

def extract_domain_name(url):
    if not url.startswith(('http://', 'https://')):
        return url
    
    parsed_url = urllib.parse.urlparse(url)
    extracted = tldextract.extract(parsed_url.netloc)
    domain_name = f"{extracted.domain}.{extracted.suffix}"

    return domain_name

def create_result_directory(output_folder, main_directory):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    final_results_path = os.path.join(script_dir, main_directory)
    new_folder_path = os.path.join(final_results_path, output_folder)

    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path, exist_ok=True)
        
    create_log_file(os.path.join(new_folder_path, "log.txt"))
    create_log_file(os.path.join(new_folder_path, "llm.txt"))
    create_log_file(os.path.join(new_folder_path, "serper.txt"))

    return {
        'log': os.path.join(new_folder_path, "log.txt"),
        'llm': os.path.join(new_folder_path, "llm.txt"),
        'serper': os.path.join(new_folder_path, "serper.txt"),
    }

def extract_year(copyright_text):
    pattern = r'\b((?:19|20)?\d{2})(?:-(\d{2}|\d{4}))?\b'

    match = re.search(pattern, copyright_text)
    
    if match:
        if match.group(2):
            return f"{match.group(1)}-{match.group(2)}"
        else:
            return match.group(1)
    else:
        return None

def get_links(url, log_file_path):
    fin = set()
    results = get_all_links(url, log_file_path)

    if isinstance(results, list):
        for item in results:
            if validators.url(item) and not is_social_media_link(item):
                fin.add(item)

    return fin

def get_all_links(url, log_file_path):
    print(url)
    max_retries = 3
    links = []

    for attempt in range(max_retries):
        with sync_playwright() as p:
            try:
                browser = p.chromium.launch(headless=False, args=['--disable-http2'])
                page = browser.new_page()
                page.goto(url, timeout=120000)
                
                page.wait_for_load_state('load')
                time.sleep(10)

                links = page.eval_on_selector_all("[href]", "elements => elements.map(el => el.href)")
                break
            except PlaywrightTimeoutError as te:
                with open(log_file_path, 'a') as f:
                    f.write(f"(Link Grabber2) Timeout error processing {url} on attempt {attempt + 1}: {te}")
                print(f"(Link Grabber2) Timeout error processing {url} on attempt {attempt + 1}: {te}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    return None
            except Exception as e:
                with open(log_file_path, 'a') as f:
                    f.write(f"(Link Grabber2) Error processing {url}: {e}")
                print(f"(Link Grabber2) Error processing {url}: {e}")
                return None
            finally:
                if page is not None:
                    page.close()
    return links