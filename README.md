# google-enhanced-llm
A simple tool designed to give chatbots more information from Google, designed to work with Ollama.


## How To Use:
1. Enable Google search API and create a search engine from Google's programmable search engine, taking note of the search engine API key and ID (This step can be easily done by following the first few minutes of this NeuralNine [video](https://www.youtube.com/watch?v=TddYMNVV14g&t=518s)).
2. Create your config(s) in the config.json file. This is where your search engine API key and ID will go.
3. Run converse.py


## Please Note:
1. This tool was originally designed to work with [Ollama](https://github.com/ollama/ollama). While an option to use models from Huggingface is included, this feature is largely untested and might produce unexpected behavior.
2. You will need to adjust the variables AMOUNT_LINKS and LINKS_LIMIT depending on how large your chosen model's context length is. A too high number for the variables will make the model forget the original question, while a too low number will not give the model sufficient information.
4. Currently, the python script only scrapes the p tag from found links.
5. While in theory this tool enables users to query a model about niche subjects or current events, please keep in mind the ability to perform this task to human standards depends largely on the chosen model's context length.


### Have Fun!
