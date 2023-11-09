import pinecone
pinecone.init(api_key="35f6e3ee-3bf3-444a-a8e6-f9824044188a", environment="gcp-starter")
index = pinecone.Index("llama-2-rag")

from torch import cuda, bfloat16
import transformers
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

model_id = "meta-llama/Llama-2-7b-chat-hf"
embed_model_id = "sentence-transformers/all-MiniLM-L6-v2"
device = f"cuda:{cuda.current_device()}" if cuda.is_available() else "cpu"

embed_model = HuggingFaceEmbeddings(
    model_name=embed_model_id,
    model_kwargs={"device": device},
    encode_kwargs={"device": device, "batch_size": 32},
)

bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16,
)

# quantization_config = transformers.BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)

hf_auth = "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"
model_config = transformers.AutoConfig.from_pretrained(model_id, use_auth_token=hf_auth)

model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    quantization_config=bnb_config,
    device_map="auto",
    use_auth_token=hf_auth,
)
model.eval()
print(f"Model loaded on {device}")

tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, use_auth_token=hf_auth)

generate_text = transformers.pipeline(
    model=model,
    tokenizer=tokenizer,
    return_full_text=True,  # langchain expects the full text
    task="text-generation",
    # we pass model parameters here too
    temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    max_new_tokens=512,  # max number of tokens to generate in the output
    repetition_penalty=1.1,  # without this output begins repeating
)

from langchain.llms import HuggingFacePipeline

llm = HuggingFacePipeline(pipeline=generate_text)

from langchain.vectorstores import Pinecone

text_field = "text"  # field in metadata that contains text content

vectorstore = Pinecone(index, embed_model.embed_query, text_field)

from langchain.chains import RetrievalQA

rag_pipeline = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever()
)


def ask_question(question):
    answer = rag_pipeline(question)
    # answer_text = answer.get("result", "No answer found")
    # if answer_text.endswith("?"):
    #     answer_text = " ".join(answer_text.split(". ")[:-1])
    #     answer_text = answer_text.split("Context:")[0].strip()
    return answer['result']


import streamlit as st

st.title("Question Answering System")

user_input = st.text_input("Ask a question:")

if user_input:
    st.write("Fetching your answer...")
    answer = ask_question(user_input)
    st.write(f"Answer: {answer}")

    user_input_1 = st.text_input('Ask a second question:')
    if user_input_1:
        st.write("Fetching your answer...")
        answer_1 = ask_question(user_input_1)
        st.write(f"Answer: {answer_1}")

        user_input_2 = st.text_input('Ask a third question:')
        if user_input_2:
            st.write("Fetching your answer...")
            answer_2 = ask_question(user_input_2)
            st.write(f"Answer: {answer_2}")

            user_input_3 = st.text_input('Ask a fourth question:')
            if user_input_3:
                st.write("Fetching your answer...")
                answer_3 = ask_question(user_input_3)
                st.write(f"Answer: {answer_3}")

                user_input_4 = st.text_input('Ask a fifth question:')
                if user_input_4:
                    st.write("Fetching your answer...")
                    answer_4 = ask_question(user_input_4)
                    st.write(f"Answer: {answer_4}")