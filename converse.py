from urllib.request import urlopen
from bs4 import BeautifulSoup
from urllib.request import urlopen
import requests, json, sys, re, time, random
from transformers import AutoTokenizer, AutoModelForCausalLM

data_file_keep = 'data.json'
url_file = 'urls.json'
convo_context = ''
responding = False
convo_counter = 0

AMOUNT_LINKS, API_KEY, SEARCH_ENGINE_ID, EXCLUDED_SITES, \
OLLAMA_MODEL, OLLAMA_API_URL, SEARCH_URL, HUGGINGFACE_REPO, LINKS_LIMIT, \
check_exclude, config_to_use = None, None, None, None, None, None, None, None, None, None, None


def set_config(new_config):
    amount_excluded = ''
    global API_KEY, SEARCH_ENGINE_ID, EXCLUDED_SITES, OLLAMA_MODEL, OLLAMA_API_URL, AMOUNT_LINKS, SEARCH_URL, HUGGINGFACE_REPO, check_exclude
    try:
        with open('config.json', 'r+') as config_file:
            print('Retrieving config...\n')
            config_data = json.load(config_file)
            configs = config_data['config']

            new_config = str(int(new_config)-1)

            configs = config_data['config'][int(new_config)]

            API_KEY = configs['API_KEY']
            SEARCH_ENGINE_ID = configs['SEARCH_ENGINE_ID']
            SEARCH_URL = configs['SEARCH_URL']
            OLLAMA_API_URL = configs['OLLAMA_API_URL']
            OLLAMA_MODEL = configs['OLLAMA_MODEL']
            HUGGINGFACE_REPO = configs['HUGGINGFACE_REPO']
            EXCLUDED_SITES = configs['EXCLUDED_SITES']
            AMOUNT_LINKS = configs['AMOUNT_LINKS']
            LINKS_LIMIT = configs['LINKS_LIMIT']

            AMOUNT_LINKS = int(AMOUNT_LINKS)
            LINKS_LIMIT = int(LINKS_LIMIT)
            
            if EXCLUDED_SITES != '' or EXCLUDED_SITES != 'none':
                check_exclude = True
            else: check_exclude = False

            amount_excluded = EXCLUDED_SITES.split(', ')

            #Set config chosen:
            config_file.seek(0)
            config_data['config_chosen'] = str(new_config)
            json.dump(config_data, config_file, indent=4)
        
        if HUGGINGFACE_REPO == '':
            print(f'Config {int(config_to_use)} pulled successfully:\n  Model: {OLLAMA_MODEL}\n  Amount links (to search for): {AMOUNT_LINKS}\n  Links limit: {LINKS_LIMIT}\n  Excluding the following sites({len(amount_excluded)}): {EXCLUDED_SITES}')
        else:   print(f'Config {int(config_to_use)} pulled successfully:\n  Huggingface Repo/model: {HUGGINGFACE_REPO}\n  Amount links (to search for): {AMOUNT_LINKS}\n  Links limit: {LINKS_LIMIT}\n  Excluding the following sites({len(amount_excluded)}): {EXCLUDED_SITES}')
    except Exception as e:
        print(f"An unexpected error occured: {e}")
        sys.exit()

def get_amount_config():
    try:
        with open('config.json', 'r') as config_file:
            config_data = json.load(config_file)
            configs = config_data['config']
        return len(configs)
    except Exception as e:
        print(f"An unexpected Error occured: {e}")
        sys.exit()

def search_engine_request(to_send, api_key, se_id, url):
    params = {
        'q': to_send,
        'key': api_key,
        'cx': se_id
    }
    response = requests.get(url, params=params)
    results = response.json()
    return results

def load_links(filename='urls.json'):
    try:
        with open(filename, 'r') as links_file:
            links_data = json.load(links_file)
    except Exception as e:
        print(f'An unexpected error occured: {e}\nIs the urls.json file empty?')
        sys.exit()
    links = links_data['links']
    return links

def scrape(all_links, tag, links_limit=None, randomize_links=False, max_chars_per_link=3400): # returns 2 things: 1. scraped_data list, should be saved.  2. data_to_give, string to give to model as data
    if len(all_links) <= 0:
        raise ValueError('No links given to scrape.')
    if len(tag) <= 0:
        raise ValueError('No tag to scrape given.')
    data_to_give = ''
    scraped_data = []
    failed, succeeded = 0, 0

    if randomize_links == True:
        random.shuffle(all_links)

    for link_info in all_links:
        link_id = link_info['id']
        link_url = link_info['link']

        if (links_limit != None) and (succeeded >= links_limit):
            print(f'Given links limit of {links_limit} hit, ending scrape...')
            break

        try:
            response = requests.get(link_url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser', from_encoding=response.encoding)
            # soup = BeautifulSoup(response.content, 'html.parser')

            paragraphs = soup.find_all(tag)
            if not paragraphs:
                print(f"WARNING: No '{tag}' tags found in link {link_id+1} ({link_url})")
                continue 
            extracted_text = ' '.join(paragraph.text for paragraph in paragraphs)
            extracted_text = extracted_text.replace('\n', '').replace('\t', '').replace('\u2013', '').replace('\u2019', '').replace('\u201d', '').replace('\u00a0', '').replace('\u201c', '')

            data_to_give += extracted_text[:max_chars_per_link]
            stored_given = extracted_text[:max_chars_per_link]

            scraped_data.append({
                'id': link_id,
                'link': link_url,
                'scraped_text': extracted_text,
                'text_given_to_model': stored_given
            })
            # all_data += extracted_text

            print(f"Scraped successfully from link {link_id+1}: {link_url}")
            succeeded += 1

        except requests.RequestException as e:
            failed += 1
            print(f"WARNING: unable to scrape data from link {link_id+1} ({link_url}): {e}")
            # no need to sys.exit() here, only a warning.
    if failed == len(all_links):
        print(f'failed: {failed} len(all_links): {len(all_links)}')
        raise ValueError('Unable to scrape data from all links.')
    else: return scraped_data, data_to_give

def ollama_call(input_query, model, ollama_api_url, data_for_model=None, context=None, gen_search=False):
    if gen_search == False:
        if context is None:
            prompt = f"Respond directly to the following prompt: {input_query}\nIf needed, use the following data to help respond:{data_for_model}\nIgnore all data irrelevant to the question.\nIf you used the data, say where in the data you got your answer from. Do not make up answers. Keep your answer short."
            data = {
                "model": model,
                "prompt": prompt
            }
        elif context is not None:
            prompt = f"Answer this follow up question: {input_query}\nIf you need to, use this data: {data_for_model}\nIf you used the data, say where in the data you got your answer from. Do not make up answers. Keep your answer short."
            data = {
                "model": model,
                "prompt": prompt,
                "context": context
            }
    else:
        prompt = f"Change the following question/statement into an effective google search using key words: {input_query}\nInclude only the search term in your response and nothing else. Don't put quotation marks in your response. Keep your response short."
        data = {
            "model": model,
            "prompt": prompt
        }

    try:
        print('Waiting for response...')
        response = requests.post(ollama_api_url, json=data)
    except Exception as e:
        print(f"An unexpected error occured: {e}. Did you forget 'ollama serve'?")
        raise

    return response

def use_huggingface_model(input_query, repo_id, data_for_model=None, gen_search=False, context=False):
    from huggingface_hub import login
    login()

    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    model = AutoModelForCausalLM.from_pretrained(repo_id, device_map="auto")
    prompt = ''

    if gen_search == False:
        if context == False:
            prompt = f"Respond directly to the following prompt: {input_query}\nIf needed, use the following data to help respond:{data_for_model}\nIgnore all data irrelevant to the question.\nIf you used the data, say where in the data you got your answer from. Do not make up answers. Keep your answer short."
        else:
            prompt = f"Answer this follow up question: {input_query}\nIf you need to, use this data: {data_for_model}\nIf you used the data, say where in the data you got your answer from. Do not make up answers. Keep your answer short."
    else:
        prompt = f"Change the following question/statement into an effective google search using key words: {input_query}\nInclude only the search term in your response and nothing else. Don't put quotation marks in your response. Keep your response short."

    input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**input_ids)
    return outputs

def process_ollama_response(response, print_out=False): # Returns nothing if print_out=True
    compiled = ''

    for line in response.iter_lines(decode_unicode=True):
        if line:
            try:
                json_data = json.loads(line)
                if json_data['done'] == False:
                    final_response = json_data['response']
                    if print_out == True:
                        print(final_response, end='')
                    else: compiled += final_response
                else: convo_context = json_data['context']
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
    if print_out == False:
        return compiled

print("Welcome to google enhanced llm search!\n")

config_to_use = input(f"Which config to use? (int, currently {get_amount_config()} configs): ")
set_config(config_to_use)
print('\nWhat would you like to know?')

# chatting sequence
while(True):
    links_list = []
    all_response = ''

    query = input(">> ")

    if HUGGINGFACE_REPO == '':
        search_query_initial = ollama_call(input_query=query, model=OLLAMA_MODEL, ollama_api_url=OLLAMA_API_URL, gen_search=True)
        search_query = process_ollama_response(search_query_initial)
    else:
        search_query_initial = use_huggingface_model(input_query=query, repo_id=HUGGINGFACE_REPO, gen_search=True)
        search_query_initial = str(search_query_initial)

    search_query = str(search_query).replace('"', '')


    with open('config.json', 'r+') as config_file:
        config_data = json.load(config_file)
        config_file.seek(0)
        config_data['prompt'] = str(query)
        json.dump(config_data, config_file, indent=4)
        config_file.truncate()

    if check_exclude == True:
        excluded_sites = EXCLUDED_SITES.split(', ')
        for i, site in enumerate(excluded_sites):
            search_query += f' -site:{site}'
    else: print('Not excluding sites, continuing...')

    print(f"Final search engine query: {search_query}\n")

    # Acquiring links
    results = search_engine_request(search_query, API_KEY, SEARCH_ENGINE_ID, SEARCH_URL)

    if 'items' in results:
        for i in range(min(AMOUNT_LINKS, len(results['items']))):
            link = results['items'][i]['link']
            links_list.append(link)

    print("Links:")
    for i, link in enumerate(links_list, start=1):
        print({link})
    print(f'{i} link(s)')

    print('Saving to Json...')
    data = {"links": []}

    for idx, link in enumerate(links_list):
        save_info = {
            "id": idx,
            "link": link
        }
        data["links"].append(save_info)

    with open('urls.json', 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=4)
    print('Done')


    # Scraping process
    links = load_links()
    if responding == True:
        scraped_results = scrape(all_links=links, tag='p', randomize_links=False, links_limit=LINKS_LIMIT, max_chars_per_link=1000)
    else: scraped_results = scrape(all_links=links, tag='p', randomize_links=False, links_limit=LINKS_LIMIT, max_chars_per_link=3400)
    scraped_data, data_to_give = scraped_results

    print('Saving scraped data')
    with open(data_file_keep, 'w') as data_file:
        json.dump(scraped_data, data_file, indent=4)

    print('')
    # llm process:
    if HUGGINGFACE_REPO == '':
        if convo_context == '':
            response = ollama_call(input_query=query, data_for_model=data_to_give, model=OLLAMA_MODEL, ollama_api_url=OLLAMA_API_URL, context=None)
        else:
            # context found
            response = ollama_call(input_query=query, data_for_model=data_to_give, model=OLLAMA_MODEL, ollama_api_url=OLLAMA_API_URL, context=convo_context)
            responding = True

        print(f'\nResponse: (Answering: {query})')
        process_ollama_response(response, print_out=True)
    else:
        if convo_counter > 0:
            response = use_huggingface_model(input_query=query, repo_id=HUGGINGFACE_REPO, data_for_model=data_to_give)
        else:
            response = use_huggingface_model(input_query=query, repo_id=HUGGINGFACE_REPO, data_for_model=data_to_give, context=False)
            convo_counter += 1
    
    print('\n')