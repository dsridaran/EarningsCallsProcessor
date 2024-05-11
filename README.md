# Earnings_Calls_Processor

Traditional stock price prediction models largely depend on historical data and market indicators, but often overlook the rich, unstructured multimedia data available from earnings calls. This project seeks to bridge this gap by providing a deep learning framework to structure data from earnings calls, both text and audio, which will provide deep insights into a company's performance.

This is a collaborative effort by team members [Dilan SriDaran](https://www.linkedin.com/in/dilansridaran/), [Maxime Wolf](https://www.linkedin.com/in/maxime-wolf/), and [Nuobei Zhang](https://www.linkedin.com/in/nuobeizhang/).

You can read the complete report for this project [here](https://maximewolf.com/assets/pdf/A_Novel_Earnings_Call_Dataset_for_Stock_Return_Prediction.pdf)

## Installation

The framework uses [Gemma-2b](https://huggingface.co/google/gemma-2b-it). Log in to this page with your Hugging Face account and accept the terms and conditions to have access to the model. Then, generate an access token and save it in a `.env` file in the root of the project as follows:

```
ACCESS_TOKEN = your_token
```

You also need to install the following package, in addition to the library listed in the `requirements.txt` file:

```
python3 -m spacy download en_core_web_sm
```

The main script is `main.py`, which you can run with the following command from the `src` directory:

```
python main.py
```

This will process all the files (transcripts and audio). 
