# Project Overview

This project is a Russian-language web application built with Streamlit that functions as an intelligent RAG (Retrieval-Augmented Generation) system for a pizza restaurant, Dodo Pizza. The application is designed to streamline the order-taking process. It utilizes a Large Language Model (LLM) that leverages the Few-Shot learning technique to interpret customer orders from a chat interface and assemble a shopping cart.

## Core Technologies

*   **Frontend:** Streamlit
*   **LLM Orchestration:** LangChain
*   **Knowledge Base:** The primary data for the RAG system is scraped from the Dodo Pizza website (`https://dodopizza.ru/moscow`).
*   **Few-Shot Learning Examples:** The `KnowledgeBase4FewShot.csv` file provides examples used to train the LLM. It contains customer queries and the corresponding product items.

## How it Works

The application provides a chat interface where a user can type their order in natural language (Russian or English). The LLM, guided by the few-shot examples, processes the request, identifies the desired products, and presents a summarized order back to the user, complete with quantities and prices.

## Project Structure

*   `app.py`: The main Streamlit application.
*   `requirements.txt`: Project dependencies.
*   `KnowledgeBase4FewShot.csv`: Data for few-shot learning.
*   `.env`: Configuration file for environment variables (e.g., API keys).
*   `README.md`: Project description.

## Building and Running

1.  **Set up your environment:**
    *   Create a `.env` file in the project root.
    *   Add your OpenAI API key to the `.env` file:
        ```
        OPENAI_API_KEY="YOUR_API_KEY"
        ```

2.  **Install dependencies:**
    ```bash
    uv pip install -r requirements.txt
    ```
3.  **Run the application:**
    ```bash
    streamlit run app.py
    ```
