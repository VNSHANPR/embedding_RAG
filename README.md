

# **TRANSFORMERS** : Basic Introduction & some fun!
<img width="800" alt="transformers_blog" src="https://github.com/VNSHANPR/embedding_RAG/assets/41034062/08f005a8-e19c-45b2-a67d-40e7e773ed86">

Notebook aims to produce a simple handson to develop understanding on the now common but important terms in the NLP world like Transformers, self attention, similarity search, RAG & Interactions with PDFs. 

*   Tokenization & Word **Embeddings** using ![Sentence transformers](https://https://www.sbert.net/) library based on Google BERT (Bidirectional Encoder Representations from **Transformers**)
  
*   **Visualize** word embeddings in 2D Space & learn basics of **Similarity search**.
  
*   Build FAISS (Facebook AI Similarity Search) Index on vectorized data and use it for Similarity search.
  
*   Build Intuition of a **Vector database** and how these embeddings can be stored.
  
*   Use embeddings generated on **Internal documents** ( to answer user questions using Large Language models e.ge OpenAI GPT 3 & a ChatGPT example.
  
*   **Retrieval Augmented Generation(RAG)**
  
*   **RAG using PDF documents** 

**Luanch Google Colab Notebook!** 

[<img src="https://github.com/VNSHANPR/embedding_RAG/assets/41034062/518a4779-7c1b-44b3-863f-b2042a5f0a99">](https://colab.research.google.com/github/VNSHANPR/embedding_RAG/blob/main/Transformers_Intro_embeddings_similarity_search_RAG.ipynb)

# High level basics on Transformers & the Self Attention Mechanism
Transformer architecture as described in the below paper has been the base of all LLM models, it was initally released for language translation (sequence to sequence model) following an encoder-decoder setup. The architecture proposed "Self attention" using attention vectors to associate each word in a sequence to other other words in the sequence leading to contextual understanding of words.

Attention is all you need

input sequence : "how are you doing today?" output sequence : "I'm am fine"

Encoder (input)

Input embeddings: Encode the input sequence (Word embeddings)
Positional Encoding: modify the input embedding by giving weightage to the position of the word in sequence.
Self Attention : Associate each individual words in a sequence to other words. (This is the job of Multi-head attention mechanism in the Transformers Architecture). Each self attention process is called a "head" and the output is a attention vector for each word in the input sequence.
Modern GPU Architecture : All the words in the sequence can be processed simultaneously using parallelization offered from the moder GPU architecture.
Decoder (output)

works on the output of the encoder layer & also the raw embeddings.
self attention mechanism uses "Masking" to focus on left side words.
Similar multi-head attention is used in the decoder
Encoding words using BERT embeddings leads to a vector of dimension 768.

Input context length is 512.(512 token at a time)

![image](https://github.com/VNSHANPR/embedding_RAG/assets/41034062/45011745-853e-402a-9e39-9e3cca88a723)

![image](https://github.com/VNSHANPR/embedding_RAG/assets/41034062/1d08a04e-d14a-4d73-90c6-39d8d1950e53)

## Visualizing embedding vector in 2D space with PCA

> Now we create a **word list** and encode them into BERT word embeddings with 768 dimensions ( Vector size 768) and then try convert them to 2 dimensions using Principal Component Analysis (PCA). This will give us a way to visualize the embedding vectors and see how they are seperate in the embedding space.


> In the output chart you see "Cat' & "Kitten" together , "houses" are farther away etc.



![image](https://github.com/VNSHANPR/embedding_RAG/assets/41034062/a2d058cf-5de7-47ca-b90a-f3b96f9eccf5)

## How embeddings look in tabular form?
You can see here the vectors enfolding from 0 - 768 , representing the input text in 768 dimensions. Scroll to the right to see the "label" column which includes the original text.

![image](https://github.com/VNSHANPR/embedding_RAG/assets/41034062/24363e32-7502-4646-a659-30151b182c34)

## Self attention in action

Our word list here contains the use of word **"cockpit"** in the context of an **Aeroplane** and also in the context of **database administration**. Similarly there are a bunch of sentences featured around **"flying an airbus"** or "engines in airbus" etc and then the last term **"flights of fantasy"** which is not related to an airbus.

World "cockpit" is used in different context so is the word "flights". The intention is to show how differen words are differentiated in the embedding space whe used in different content.

![image](https://github.com/VNSHANPR/embedding_RAG/assets/41034062/cada1153-0a86-4f6f-844b-0c49d5e368e1)

![image](https://github.com/VNSHANPR/embedding_RAG/assets/41034062/54e0b40f-afdd-419e-8c1c-d0f3fc0ce6e0)

## **Similarity Search example**

As we have seen we are able to represent **words & sentences**  using sentence transformers into vectors of length 768 which includes the contextual meaning of sentences as well (thanks for the tranformer architecture). With our text represented in such a way we can compare different sentences using similarity measures like **cosine similarity** or **dot product** etc.

# **Retrieval Augmented Generation (RAG)**

RAG was proposed by researchers from Meta : https://arxiv.org/pdf/2005.11401.pdf

We will use a baby example using similarity search with BERT created embeddings on a question/response dataset generated using GPT, which will act as our enterprise context along with LLM (GPT 3.5) to respond to new queries.

As a first step let's generate a dataset using GPT, a sample questions/answers set answered in a Request for Proposal (RFP) context.


## Generate a dataset using GPT

You can paste the below prompt in chatGPT to get a sample response.

Generate a dataset in csv form with 100 rows about the capabilities of a product called l2w online analytics platform.
The dataset has 4 columns : Question, Answer, RFP_doc_ID abd Date.
Populate the Question field with unique questions about the product like "how is encryption handled by the application?" and the answer field with matching responses like "product uses TLS1.2 to encrypt the data transmission". Other examples of the questions are as below :
What are the access controls in the application?

![image](https://github.com/VNSHANPR/embedding_RAG/assets/41034062/75372644-d2de-4d36-b69f-de95b0cdbc65)

## Generate embeddings on the merged text

## Store embeddings in a FAISS (Facebook AI Similarity Search) Index for Similarity search

This task can be performed by a **Vector Database** in a production setup!

## Create a Function to Retrieve Matching responses

## Trigger a **Query** & get similar documents

## Create Prompt with retreived information from Internal document

## use the prompt to get an answer from GPT openai API Playground App.

![image](https://github.com/VNSHANPR/embedding_RAG/assets/41034062/41312b6d-eefd-4c45-b662-6e9ee68f55a9)

# Interact with PDF documents

We are here interacting with a Singapore Tax reference document.

https://www.iras.gov.sg/media/docs/default-source/uploadedfiles/pdf/guide-to-form-b1-(english).pdf?sfvrsn=5c44ab37_12

First step here is to load the PDF and split into chunks of 512 word length.
This is done because the input context lenght of a BERT model is 512. After running the next code cell the PDF will be loaded and split into 71 pieces of length 512.

## Answer user queries using the information from PDF

![image](https://github.com/VNSHANPR/embedding_RAG/assets/41034062/19656090-5fe0-4e07-9dbb-64b570912200)





