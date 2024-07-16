# Wine Business Chatbot

This project implements an AI-powered chatbot for a wine business using retrieval-augmented generation. The chatbot can answer customer queries about wines, winemakers, and related information based on a provided corpus. 

## Chatbot Demo Video: https://drive.google.com/file/d/1Y1MruBXIoKeoSFjcdGOynerVvrlJFRDh/view?usp=sharing

## Chatbot Web App: https://wine-business-chatbot.streamlit.app/


## Features

- PDF text extraction and chunking
- Vector database (ChromaDB) for efficient information retrieval
- Integration with Hugging Face and Google's Generative AI (Gemini) models
- Streamlit-based web interface for easy interaction
- Context-aware responses using chat history


## Requirements

- Python 3.7+
- Streamlit
- ChromaDB
- Hugging Face Transformers
- Google Generative AI
- PDFMiner
- spaCy
- LangChain


## Installation

1. Clone this repository:
```
git clone https://github.com/yourusername/wine-business-chatbot.git
cd wine-business-chatbot
```

2. Install the required packages:
```
pip install -r requirements.txt
```

3. Download the spaCy English model:
```
python -m spacy download en_core_web_sm
```

4. Set up your environment variables or Streamlit secrets for API keys:
- `hf_token`: Hugging Face API token
- `gemini_pro_key`: Google Gemini Pro API key


## Usage

You can interact with the Wine Business Chatbot in two ways:

1. Visit the live website:
   
   The chatbot is deployed and accessible at: https://wine-business-chatbot.streamlit.app/
   
   Simply open this link in your web browser to start using the chatbot immediately.

2. Run locally:

   If you want to run the application on your local machine:

Run the Streamlit app:
```
streamlit run app.py
```

Navigate to the provided local URL in your web browser to interact with the chatbot.


## Project Structure

- `app.py`: Main application file containing the Streamlit interface and chatbot logic
- `Corpus.pdf`: Source PDF containing wine business information
- `db/`: Directory for storing the ChromaDB database
- `requirements.txt`: List of Python dependencies


## Future Enhancements

- Multi-modal capabilities for image processing
- User personalization
- Appointment booking functionality
- Real-time inventory integration
- Voice interface
- Multi-language support
- Analytics dashboard
- E-commerce integration
- Wine recommendation system
- Social media sharing features


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
