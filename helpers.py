from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
import config
import time
import validators
import re

social_media_domains = [
    'facebook.com', 'twitter.com', 'instagram.com', 'cookiepedia.co.uk', 'fonts.googleapis.com', 'jwt.io', 'google-analytics.com', 'adobe.com',
    'threads.net', 'linkedin.com', 'pinterest.com', 'youtube.com', 'onetrust.com', 'amazon.com', 'reddit.com', 'wordpress.org', 'adobe.io',
    'tiktok.com', 'snapchat.com', 'whatsapp.com', 'quora.com', 'google.com', 'github.com', 'apple.com', 'vimeo.com', 'youtu.be', 'cloudflare.net', 'goo.gl', 'mozilla.org', 'maps.app.goo.gl'
]

def is_social_media_link(link):
    return any(domain in link for domain in social_media_domains)

def set_log_file(file_path):
    with open(file_path, 'w') as f:
        f.write('')

    config.log_file = file_path

def get_log_file():
    return config.log_file

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

def get_links(url):
    fin = set()
    results = get_all_links(url)

    if isinstance(results, list):
        for item in results:
            if validators.url(item) and not is_social_media_link(item):
                fin.add(item)

    return fin

def get_all_links(url):
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
                # with open(get_log_file(), 'a') as f:
                #     f.write(f"(Link Grabber2) Timeout error processing {url} on attempt {attempt + 1}: {te}")
                print(f"(Link Grabber2) Timeout error processing {url} on attempt {attempt + 1}: {te}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    return None
            except Exception as e:
                # with open(get_log_file(), 'a') as f:
                #     f.write(f"(Link Grabber2) Error processing {url}: {e}")
                print(f"(Link Grabber2) Error processing {url}: {e}")
                return None
            finally:
                if page is not None:
                    page.close()
    return links