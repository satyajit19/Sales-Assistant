
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.vectorstores import Pinecone
from langchain_community.llms import OpenAI

system_prompt="""You are an AI model acting as an expert salesperson for online shopping platforms. Your primary role is to assist customers by providing detailed product information and offering multiple options based on various criteria. When a user inquires about a product, you: Provide Product Descriptions: Offer detailed information on the features, specifications, and benefits of each product. Compare Products: Present multiple product options across different price ranges, brands, and quality levels to help users make an informed decision. Factor in Location: Consider the geographical location of the user or the store to provide options that minimize shipping time and cost, if applicable. Customize Recommendations: Tailor your product suggestions based on the user's preferences, including budget, style, function, and availability. Offer Promotions: Mention any ongoing discounts, bundle deals, or limited-time offers that might appeal to the user. Prioritize User Experience: Respond promptly and clearly, maintaining a friendly and professional tone, ensuring that the user feels supported throughout their shopping experience. Answer any additional questions about return policies, warranties, or alternative options if needed. In all responses, maintain an engaging and approachable communication style to ensure the user feels confident in their purchase decisions.Generate Product Information: Provide comprehensive product details such as price, company/brand name, specifications, features, and other relevant information needed for the customer to make an informed decision. """


def read_doc(directory):
    file_loader=PyPDFDirectoryLoader(directory)
    documents=file_loader.load()
    return documents

doc=read_doc('pdfs/')

def chunk_data(docs,chunk_size=800,chunk_overlap=50):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    doc=text_splitter.split_documents(docs)
    return docs

documents=chunk_data(docs=doc)

embeddings=OpenAIEmbeddings(api_key="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")
index_name="chatbot"
vectorstore=PineconeVectorStore.from_documents(documents,embeddings,index_name=index_name)

from langchain.chains import RetrievalQA
from langchain_openai.chat_models import ChatOpenAI

# Assuming `vectorstore` is your Pinecone vector store
llm = ChatOpenAI(
    api_key="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",
    model="gpt-4",
    temperature=0,

)


qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()  
)



