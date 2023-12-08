# DataSpeak Question-Answering Chatbot

## Project Overview
Developed a prototype Question-Answering chatbot for DataSpeak to enhance customer service efficiency. This chatbot allows customers to ask questions and receive generative responses, using a unique retrieval-augmented generation approach.

## Technology Stack
- **Data Processing:** Google Colab with GPU access for preprocessing question-answer data and generating embeddings.
- **Indexing:** Pinecone for upserting and managing embeddings.
- **Language Model:** Llama 2 LLM from Hugging Face.
- **Pipeline Integration:** LangChain's RetrievalQA tool combined with the LLM for effective question-answering.
- **Web Application:** Streamlit for user interface and demonstration.

## Project Methodology
- Embeddings from question-answer data were processed and stored in Pinecone.
- The Llama 2 LLM was integrated for generating responses.
- A pipeline using LangChain's RetrievalQA tool retrieved relevant data from the Pinecone index to aid the LLM in answer generation.
- The entire process was encapsulated in a Streamlit web application for user interaction.

## Key Features
- **Generative Responses:** Chatbot generates answers using contextually relevant data retrieved from the company’s database.
- **Efficient Data Retrieval:** Utilizes Pinecone for fast and accurate retrieval of matching question-answer pairs.
- **User-Friendly Interface:** Streamlit-based web application for easy interaction with the chatbot.

## Installation and Deployment
- Open `app_launcher.ipynb` in the Google Collab interface.
- Upload `app.py` as a temporary file in the workspace.
- Follow the instructions in `app_launcher.ipynb` to access your own authorization code for the hosting service used.
- Run `app_launcher.ipynb`, enabling GPU utilization. 

## Acknowledgments
- This project was part of the #TripleTen data science bootcamp/certificate program, serving as an externship.
- Special thanks to C-level staff at DataSpeak and my fellow students for their support and collaboration.

## Presentation Slides
[[https://github.com/jnorfolk/DataSpeak-QA/Dataspeak-QA-Project.pptx](https://github.com/jnorfolk/DataSpeak-QA/blob/main/Dataspeak%20QA%20Project.pptx)](https://view.officeapps.live.com/op/view.aspx?src=https%3A%2F%2Fraw.githubusercontent.com%2Fjnorfolk%2FDataSpeak-QA%2Fmain%2FDataspeak%2520QA%2520Project.pptx&wdOrigin=BROWSELINK)https://view.officeapps.live.com/op/view.aspx?src=https%3A%2F%2Fraw.githubusercontent.com%2Fjnorfolk%2FDataSpeak-QA%2Fmain%2FDataspeak%2520QA%2520Project.pptx&wdOrigin=BROWSELINK
