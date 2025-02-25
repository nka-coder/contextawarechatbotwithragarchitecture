from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

# Load environment variables from .env file
load_dotenv()

# Step 1: Load and preprocess documents from multiple files
def load_and_preprocess_documents(file_paths):
    documents = []
    
    for file_path in file_paths:
        if file_path.endswith(".txt"):
            loader = TextLoader(file_path)
            documents.extend(loader.load())
        elif file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
        else:
            print(f"Unsupported file format: {file_path}")
    
    # Split the documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    
    return texts

# Step 2: Create a vector store for document retrieval
def create_vector_store(texts):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(texts, embeddings)
    return vector_store

# Step 3: Set up the Retrieval-Augmented Generation (RAG) chain
def create_rag_chain(vector_store):
    llm = OpenAI(temperature=0.7)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        return_source_documents=False
    )
    return qa_chain

# Step 4: Create a context-aware chatbot
def chatbot(qa_chain):
    print("Hello I'm a Customer Support Chatbot!")
    print("Type 'exit' to end the conversation.")
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() == "exit":
            print("Chatbot: Goodbye!")
            break
        
        # Get the response from the RAG chain
        response = qa_chain({"query": user_input})
        
        # Display the chatbot's response
        print(f"Chatbot: {response['result']}")
        
        # Optionally, display the source documents for reference
        # if response['source_documents']:
            # print("\nSources:")
            # for doc in response['source_documents']:
                # source = doc.metadata.get('source', 'Unknown')
                # print(f"- {source}: {doc.page_content[:200]}...")  # Display first 200 chars

# Main function to run the chatbot
def main():
    # Paths to your knowledge base documents (can be .txt or .pdf)
    file_paths = [
        "docs/dell.pdf",
        "docs/canon.pdf"
    ]
    
    # Load and preprocess the documents
    texts = load_and_preprocess_documents(file_paths)
    
    # Create a vector store
    vector_store = create_vector_store(texts)
    
    # Create the RAG chain
    qa_chain = create_rag_chain(vector_store)
    
    # Run the chatbot
    chatbot(qa_chain)

if __name__ == "__main__":
    main()