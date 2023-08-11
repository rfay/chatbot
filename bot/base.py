from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
model = GPT2LMHeadModel.from_pretrained("gpt2-medium")

from bs4 import BeautifulSoup
import requests

def scrape_website(url):
    # Fetch the content from url
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')

    # Extract the text from the webpage
    text = soup.get_text()

    return text


def generate_response(input_text):
    # check if the input text is a URL
    if 'http' in input_text or 'www' in input_text:
        scraped_text = scrape_website(input_text)
        input_text += " " + scraped_text

    # encode the input text
    input_text = tokenizer.encode(input_text, return_tensors='pt')

    # generate a response
    response = model.generate(input_text, max_length=2000, pad_token_id=tokenizer.eos_token_id, num_return_sequences=1)

    # decode the response
    response_text = tokenizer.decode(response[0])

    return response_text


from flask import Flask, request

app = Flask(__name__)

# @app.route('/bot', methods=['POST'])
@app.route('/', methods=['GET', 'POST'])

def bot():
    data = request.get_json()
    message = data['message']
    response = generate_response(message)
    return {'message': response}

if __name__ == '__main__':
    app.run(port=8000, debug=True)
