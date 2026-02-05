import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_core.prompts import PromptTemplate

load_dotenv()

def main():
    st.title("Dodo Pizza Bot")

    openai_api_key='sk-al8Dzmdmj-xJy13SoUR5aQ'
    embeddings = OpenAIEmbeddings(api_key=openai_api_key, model='text-embedding-3-small', 
                                        base_url="https://api.vsellm.ru/")
    db = FAISS.load_local("notebooks/dodo_faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever()

    template = """
You are a helpful assistant for Dodo Pizza. Your task is to process a user's order and present it in a clear, itemized format.

Use the provided context, which contains information about products and prices, to fulfill the order.

The user's order must be formatted exactly as in the example below.
You must extract the product name, quantity, and price from the context.
You must calculate the total price for the order.

**EXAMPLE:**
User request: "Можно мне сок "Добрый" и две маленьких пиццы "Пепперони"?"
Your response:
Ваш заказ:
- сок "Добрый" - 1 х 150 рублей
- пицца "Пепперони" (25 см) - 2 х 350 рублей
Итого: 850 рублей

---
Now, process the following request.

Use the following pieces of context to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

User request: {question}
Your response:
"""
    
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    
    memory = ConversationBufferWindowMemory(k=5, memory_key="chat_history", return_messages=True)
    
    llm = ChatOpenAI(api_key=openai_api_key, model='gpt-4o-mini',
                        base_url="https://api.vsellm.ru/")  
    
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        verbose=True
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_prompt := st.chat_input("What would you like to order?"):
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            response = qa.invoke({"question": user_prompt})
            st.markdown(response["answer"])
        st.session_state.messages.append({"role": "assistant", "content": response["answer"]})

if __name__ == "__main__":
    main()
