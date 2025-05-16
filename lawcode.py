import streamlit as st
import pymongo
import bcrypt
import time
import datetime
import re
from pythainlp.tokenize import word_tokenize
from bson import ObjectId
from functools import lru_cache
from numpy import dot
from numpy.linalg import norm
from datetime import datetime
from uuid import uuid4
import itertools
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv


load_dotenv()
HF_TOKEN = os.environ["HF_TOKEN"]

@st.cache_resource
def get_llm():
    tokenizer, model = get_tokenizer_and_model()
    return pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

@st.cache_resource
def get_embedder():
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

@st.cache_resource
def get_tokenizer_and_model():
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        token=HF_TOKEN
    )
    return tokenizer, model

# ใช้ global ตัวแปร
tokenizer, model = get_tokenizer_and_model()

# MongoDB Atlas Connection
MONGO_URI = "mongodb+srv://witsawadaochui:aGTd8WWVkzpxXaC5@project0.9uela.mongodb.net/?retryWrites=true&w=majority&appName=Project0"
client = pymongo.MongoClient(MONGO_URI)
db = client["law_database"]
document_collection = db["documents"]
conversation_collection = db["conversations"]



# Create an empty placeholder for the alert message
alert_placeholder = st.empty()


# Function to hash passwords (if needed in the future)
def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())


# ฟังก์ชันคำนวณความคล้ายคลึงของคอไซน์
def cosine_similarity(vec1, vec2):
    """Calculate the cosine similarity between two vectors."""
    return dot(vec1, vec2) / (norm(vec1) * norm(vec2))


# ฟังก์ชั่นนี้จะให้จำนวนคำทั้งหมดในข้อความที่สามารถนับได้
def count_tokens(text):
    tokens = text.split()
    return len(tokens)


# Fetch chat history from MongoDB
def get_chat_history(username):
    try:
        chats = list(conversation_collection.find({"username": username}).sort("timestamp", -1))
        return chats
    except Exception as e:
        st.error(f"Error fetching chat history: {str(e)}")
        return []


# Message----------------------------------------------------------------------------------------------------------------------------Message

# เพิ่ม CSS ให้ปุ่มใน sidebar ยืดเต็มความกว้าง
def add_css():
    st.markdown("""
        <style>
        .stButton > button {
            width: 100%;
        }
        </style>
    """, unsafe_allow_html=True)


# Message

def display_ai_message_gradually(ai_response, message_placeholder):
    # If the response is not a DataFrame, display text gradually
    displayed_text = ""
    for char in ai_response:
        displayed_text += char
        message_placeholder.markdown(f"{displayed_text}")
        time.sleep(0.01)  # Adjust the speed here (0.01 = 10ms delay per character)


# ฟังก์ชันแสดงบทสนทนาที่ถูกเลือกใน Main Area เท่านั้น (ปรับให้แสดงคำถามและคำตอบหลายชุด)
# Display selected chat in main page with animations
# def show_selected_chat_in_main_area():
#     if "selected_chat" in st.session_state and st.session_state["selected_chat"]:
#         chat = st.session_state["selected_chat"]
#         main_area_placeholder = st.empty()
#         with main_area_placeholder.container():
#             for message in st.session_state["messages"]:
#                 role = message["role"]
#                 content = message["content"]
#                 if role == "user":
#                     st.chat_message("user", content)
#                 elif role == "assistant":
#                     message_placeholder = st.empty()
#                     display_typing_animation(message_placeholder, duration=2)
#                     displayed_text = ""
#                     for char in content:
#                         displayed_text += char
#                         message_placeholder.markdown(f"{displayed_text}")
#                         time.sleep(0.01)
#     else:
#         st.write("No chat selected.")


# Message

# Show chat history in sidebar
def show_chat_history_in_sidebar(username):
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = get_chat_history(username)

    chat_history = st.session_state['chat_history']
    st.sidebar.subheader(f"Chat History ({username})")

    for chat in chat_history:
        col1, col2 = st.sidebar.columns([8, 2])
        with col1:
            if col1.button(f"Q: {chat['question'][:30]}...", key=str(chat["_id"])):
                st.session_state["selected_chat"] = chat
        with col2:
            if col2.button("Delete", key=f"delete-{chat['_id']}"):
                delete_conversation(chat["_id"])


# Message_END----------------------------------------------------------------------------------------------------------------------------Message_END


# ฟังก์ชันนี้ช่วยให้สามารถลบบทสนทนาทั้งในฐานข้อมูลและในแอปพลิเคชัน
def delete_conversation(conversation_id):
    conversation_collection.delete_one({"_id": ObjectId(conversation_id)})
    st.session_state['chat_history'] = [chat for chat in st.session_state['chat_history'] if
                                        str(chat["_id"]) != str(conversation_id)]
    st.sidebar.success("Deleted conversation successfully!")


# ฟังก์ชันนี้ช่วยเพิ่มประสิทธิภาพในการเรียกข้อมูล embedding โดยเฉพาะเมื่อมีการประมวลผลข้อความซ้ำบ่อย ๆ
@lru_cache(maxsize=128)
def get_embedding_cache(question):
    return bedrock_embeddings.embed_query(question)


# ฟังก์ชัน get_unique_id ใช้สำหรับสร้าง ID ที่ไม่ซ้ำกัน โดยใช้ ObjectId จาก MongoDB
def get_unique_id():
    return str(ObjectId())


# ฟังก์ชัน clean_and_tokenize นี้ใช้สำหรับทำความสะอาดและแยกคำ (tokenize) จากข้อความที่ได้รับ (content)
def clean_and_tokenize(content):
    # ลบช่องว่างที่ต่อกันมากกว่า 1 ช่องให้เหลือเพียง 1 ช่อง
    cleaned_content = re.sub(r'\s+', ' ', content).strip()
    # แก้ไขปัญหาตัวอักษร 'ำ' ที่ถูกแทนเป็น 'ำา'
    cleaned_content = cleaned_content.replace('ำา', 'ำ')
    # tokenize ข้อความ
    tokens = word_tokenize(cleaned_content, keep_whitespace=False)
    # ลบช่องว่างทั้งหมดออกจากข้อความ
    return "".join(tokens)  # รวมข้อความโดยไม่ใส่ช่องว่าง


# ฟังก์ชัน delete_files_from_s3_and_mongodb ใช้สำหรับลบไฟล์จาก Amazon S3 และเอกสารที่เกี่ยวข้องจาก MongoDB
def delete_files_from_s3_and_mongodb(bucket_name, selected_file_names):
    for file_name in selected_file_names:
        for file in selected_file_names[file_name]:
            try:
                # # ลบไฟล์จาก S3
                # s3_client.delete_object(Bucket=bucket_name, Key=file)
                # st.write(f"Deleted file from S3: {file}")
                # ลบเอกสารจาก MongoDB โดยใช้ชื่อไฟล์
                result = document_collection.delete_many({"pdf_name": file_name})
                if result.deleted_count > 0:
                    st.write(f"Deleted {result.deleted_count} documents from MongoDB for file: {file_name}")
                else:
                    st.write(f"No documents found in MongoDB for file: {file_name}")
            except Exception as e:
                st.write(f"Error deleting file {file}: {str(e)}")

# การถามทั้งหมด--------------------------------------------------------------------------------------------------------------------------------------------การถามทั้งหมด

# ฟังก์ชันสำหรับ Tokenization
def tokenize_text(text):
    if not text:
        raise ValueError("ข้อความว่างเปล่า")
    tokens = tokenizer.tokenize(text)
    if not tokens:
        raise ValueError("ไม่สามารถแปลงข้อความเป็น token ได้")
    return tokens


# ฟังก์ชันสำหรับจำแนกประเภทคำถาม
def classify_question_type(question):
    prompt = f"จงจำแนกคำถามนี้ว่าเป็นประเภทใด (0=มาตรา, 1=ความหมาย, 2=แนวทาง, 3=ไม่เกี่ยวกับกฎหมาย):\n\nคำถาม: {question}\n\nประเภท:"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    output = model.generate(input_ids, max_new_tokens=10)
    predicted_class_text = tokenizer.decode(output[0], skip_special_tokens=True).strip()

    # พยายามดึงเลขคลาสจากข้อความ เช่น "ประเภท: 1"
    match = re.search(r"\b([0-3])\b", predicted_class_text)
    return int(match.group(1)) if match else 3  # ถ้าไม่ตรง ส่งว่าไม่เกี่ยวกับกฎหมาย



# # ฟังก์ชันสร้าง prompt ตามประเภทคำถามที่ครอบคลุมมากขึ้น
# def generate_prompt_based_on_type(question, question_type):
#     if question_type == 0:
#         # ประเภท 0: ถามเกี่ยวกับมาตราและความผิด
#         prompt = f"""
#         ขยายความคำถามนี้ให้ละเอียดขึ้นโดยไม่ต้องสร้างคำตอบไม่เกิน 500 Token โดยให้เพิ่มรายละเอียดที่จำเป็นเพื่อให้เข้าใจได้ชัดเจนยิ่งขึ้นในเชิงกฎหมาย:
#         - อธิบายการกระทำที่อาจละเมิดกฎหมายอย่างชัดเจน รวมถึงประเภทของการกระทำที่เกี่ยวข้อง (เช่น การลักทรัพย์, การทำร้ายร่างกาย, หรือการละเมิดสิทธิต่างๆ)
#         - ระบุถึงมาตราหรือบทกฎหมายที่ควรจะใช้พิจารณาการกระทำนี้ พร้อมทั้งอธิบายเนื้อหาของกฎหมายที่เกี่ยวข้องและสาเหตุที่ใช้มาตรานั้น ๆ
#         - เพิ่มข้อมูลเกี่ยวกับบทลงโทษที่อาจเกิดขึ้นตามกฎหมาย รวมถึงตัวเลือกของบทลงโทษที่อาจใช้ตามสภาพความรุนแรงหรือสภาพแวดล้อมของการกระทำ
#         - หากมีข้อยกเว้นหรือเงื่อนไขที่ทำให้การกระทำนี้ไม่ถือเป็นความผิด (เช่น การป้องกันตัวเองหรือการกระทำที่ไม่ตั้งใจ) โปรดระบุด้วย และอธิบายถึงเงื่อนไขหรือสถานการณ์ที่ข้อยกเว้นเหล่านี้สามารถนำมาใช้ได้
#
#         คำถาม: '{question}'
#         """
#
#     elif question_type == 1:
#         # ประเภท 1: ถามเกี่ยวกับความหมาย
#         prompt = f"""
#         ขยายความคำถามนี้ให้ละเอียดขึ้นโดยไม่ต้องสร้างคำตอบไม่เกิน 500 Token โดยเน้นถึงความหมายและการตีความในเชิงกฎหมายของคำหรือแนวคิดที่สำคัญ:
#         - อธิบายคำหรือแนวคิดที่อาจก่อให้เกิดความสับสนหรือมีหลายความหมายในเชิงกฎหมาย พร้อมทั้งเน้นความหมายที่เกี่ยวข้องกับคำถามนี้เป็นพิเศษ
#         - เพิ่มตัวอย่างการใช้คำดังกล่าวในบริบททางกฎหมาย เช่น การใช้ในศาลหรือการบังคับใช้กฎหมาย เพื่อช่วยให้เห็นภาพการใช้งานจริง
#         - เพิ่มการอ้างอิงถึงมาตราหรือข้อกำหนดที่เกี่ยวข้อง โดยระบุว่าแนวคิดดังกล่าวมีการระบุไว้อย่างไรในกฎหมาย และอธิบายถึงการตีความที่อาจแตกต่างกันในสถานการณ์ต่าง ๆ
#         - ระบุถึงข้อถกเถียงหรือการตีความในเชิงกฎหมายที่อาจเกิดขึ้นจากแนวคิดนี้ เพื่อให้เห็นถึงความหลากหลายของมุมมอง
#
#         คำถาม: '{question}'
#         """
#
#     elif question_type == 2:
#         # ประเภท 2: ถามแนวทางปฏิบัติ
#         prompt = f"""
#         ขยายความคำถามนี้ให้ละเอียดขึ้นโดยไม่ต้องสร้างคำตอบไม่เกิน 500 Token เพื่ออธิบายถึงแนวทางปฏิบัติตามกฎหมายในบริบทที่กว้างขึ้น โดยให้ครอบคลุมการปฏิบัติตามกฎหมายและการป้องกันการละเมิดกฎหมาย:
#         - เพิ่มรายละเอียดของขั้นตอนการปฏิบัติตามกฎหมายในแต่ละขั้นตอน ตั้งแต่การประเมินสถานการณ์ไปจนถึงการดำเนินการตามข้อกำหนดทางกฎหมาย
#         - อธิบายวิธีการติดต่อหรือขั้นตอนในการประสานงานกับหน่วยงานทางกฎหมายที่เกี่ยวข้อง เช่น กระบวนการแจ้งความ การยื่นฟ้อง การป้องกันตนเอง หรือการรายงานเหตุการณ์
#         - ระบุถึงกฎหมายและมาตราที่เกี่ยวข้องกับสถานการณ์นี้ พร้อมทั้งให้รายละเอียดถึงแนวทางที่กฎหมายแนะนำว่าควรปฏิบัติอย่างไรในสถานการณ์นี้
#         - เพิ่มคำแนะนำในการป้องกันการกระทำที่อาจเข้าข่ายผิดกฎหมาย เพื่อให้บุคคลสามารถปฏิบัติตามกฎหมายได้อย่างถูกต้องและปลอดภัย
#
#         คำถาม: '{question}'
#         """
#
#     return prompt
#
#
# def generate_answer_prompt_based_on_type(expanded_question, question_type, top_docs):
#     # สำหรับประเภทคำถามที่ 0: ถามเกี่ยวกับมาตราและความผิด
#     if question_type == 0:
#         answer_prompt = f"""
#         คุณกำลังจะตอบคำถามเกี่ยวกับมาตราและความผิด โปรดให้คำตอบที่ชัดเจนและครอบคลุม โดย:
#         - อธิบายการกระทำที่เกี่ยวข้องกับมาตราและความผิดอย่างละเอียด
#         - ระบุบทกฎหมายที่เกี่ยวข้อง พร้อมอธิบายเหตุผลที่ใช้มาตรานั้นๆ
#         - หากมีบทลงโทษที่อาจเกิดขึ้น ให้ระบุบทลงโทษตามกฎหมาย
#         - ถ้ามีข้อยกเว้นที่ทำให้การกระทำนี้ไม่ถือเป็นความผิด เช่น การป้องกันตัวเอง ให้ระบุข้อยกเว้นนั้น
#
#         คำถามที่ขยาย: '{expanded_question}'
#         เนื้อหาที่เกี่ยวข้อง: {top_docs}
#         """
#
#     # สำหรับประเภทคำถามที่ 1: ถามเกี่ยวกับความหมาย
#     elif question_type == 1:
#         answer_prompt = f"""
#         คุณกำลังจะตอบคำถามเกี่ยวกับความหมายและการตีความในเชิงกฎหมาย โปรดให้คำตอบที่ชัดเจนและครบถ้วน โดย:
#         - อธิบายคำหรือแนวคิดในเชิงกฎหมายที่ชัดเจน
#         - ยกตัวอย่างการใช้งานในบริบทกฎหมาย เช่น การตีความในศาล
#         - หากมีการตีความต่างๆ ตามบริบทต่างๆ เช่น ศาลหรือตำรวจ ให้ระบุด้วย
#
#         คำถามที่ขยาย: '{expanded_question}'
#         เนื้อหาที่เกี่ยวข้อง: {top_docs}
#         """
#
#     # สำหรับประเภทคำถามที่ 2: ถามแนวทางปฏิบัติ
#     elif question_type == 2:
#         answer_prompt = f"""
#         คุณกำลังจะตอบคำถามเกี่ยวกับแนวทางปฏิบัติตามกฎหมาย โปรดให้คำตอบที่ชัดเจน โดย:
#         - อธิบายขั้นตอนปฏิบัติตามกฎหมายในสถานการณ์นี้ เช่น การแจ้งความ การยื่นฟ้อง
#         - ระบุถึงกฎหมายและมาตราที่เกี่ยวข้อง เพื่อให้ผู้ถามปฏิบัติตามได้อย่างถูกต้อง
#
#         คำถามที่ขยาย: '{expanded_question}'
#         เนื้อหาที่เกี่ยวข้อง: {top_docs}
#         """
#
#     return answer_prompt


def generate_prompt(question, question_type, purpose="expand", top_docs=None):
    max_token_count = 500  # กำหนด token สูงสุดให้กับ prompt

    # ถ้ามีเอกสารที่เกี่ยวข้อง ให้รวมเอกสารเป็นส่วนหนึ่งของ prompt
    context_content = ""
    if top_docs:
        context_parts = []
        for doc, score in top_docs:
            content = doc.get("chunk_content", "")
            context_parts.append(f"เนื้อหาอ้างอิง: {content}")
        context_content = "\n\n".join(context_parts)

    if purpose == "expand":
        # Prompt สำหรับการขยายคำถาม
        if question_type == 0:
            prompt = f"""
                    ขยายความคำถามนี้ให้ละเอียดขึ้นโดยไม่ต้องสร้างคำตอบไม่เกิน 500 Token โดยให้เพิ่มรายละเอียดที่จำเป็นเพื่อให้เข้าใจได้ชัดเจนยิ่งขึ้นในเชิงกฎหมาย:
                    - อธิบายการกระทำที่อาจละเมิดกฎหมายอย่างชัดเจน รวมถึงประเภทของการกระทำที่เกี่ยวข้อง (เช่น การลักทรัพย์, การทำร้ายร่างกาย, หรือการละเมิดสิทธิต่างๆ)
                    - ระบุถึงมาตราหรือบทกฎหมายที่ควรจะใช้พิจารณาการกระทำนี้ พร้อมทั้งอธิบายเนื้อหาของกฎหมายที่เกี่ยวข้องและสาเหตุที่ใช้มาตรานั้น ๆ
                    - เพิ่มข้อมูลเกี่ยวกับบทลงโทษที่อาจเกิดขึ้นตามกฎหมาย รวมถึงตัวเลือกของบทลงโทษที่อาจใช้ตามสภาพความรุนแรงหรือสภาพแวดล้อมของการกระทำ
                    - หากมีข้อยกเว้นหรือเงื่อนไขที่ทำให้การกระทำนี้ไม่ถือเป็นความผิด (เช่น การป้องกันตัวเองหรือการกระทำที่ไม่ตั้งใจ) โปรดระบุด้วย และอธิบายถึงเงื่อนไขหรือสถานการณ์ที่ข้อยกเว้นเหล่านี้สามารถนำมาใช้ได้

                    คำถาม: '{question}'
                    """
            if context_content:
                prompt += f"\n\nข้อมูลอ้างอิงที่เกี่ยวข้อง:\n{context_content}"

        elif question_type == 1:
            prompt = f"""
                    ขยายความคำถามนี้ให้ละเอียดขึ้นโดยไม่ต้องสร้างคำตอบไม่เกิน 500 Token โดยเน้นถึงความหมายและการตีความในเชิงกฎหมายของคำหรือแนวคิดที่สำคัญ:
                    - อธิบายคำหรือแนวคิดที่อาจก่อให้เกิดความสับสนหรือมีหลายความหมายในเชิงกฎหมาย พร้อมทั้งเน้นความหมายที่เกี่ยวข้องกับคำถามนี้เป็นพิเศษ
                    - เพิ่มตัวอย่างการใช้คำดังกล่าวในบริบททางกฎหมาย เช่น การใช้ในศาลหรือการบังคับใช้กฎหมาย เพื่อช่วยให้เห็นภาพการใช้งานจริง
                    - เพิ่มการอ้างอิงถึงมาตราหรือข้อกำหนดที่เกี่ยวข้อง โดยระบุว่าแนวคิดดังกล่าวมีการระบุไว้อย่างไรในกฎหมาย และอธิบายถึงการตีความที่อาจแตกต่างกันในสถานการณ์ต่าง ๆ
                    - ระบุถึงข้อถกเถียงหรือการตีความในเชิงกฎหมายที่อาจเกิดขึ้นจากแนวคิดนี้ เพื่อให้เห็นถึงความหลากหลายของมุมมอง

                    คำถาม: '{question}'
                    """
            if context_content:
                prompt += f"\n\nข้อมูลอ้างอิงที่เกี่ยวข้อง:\n{context_content}"

        elif question_type == 2:
            prompt = f"""
                   ขยายความคำถามนี้ให้ละเอียดขึ้นโดยไม่ต้องสร้างคำตอบไม่เกิน 500 Token เพื่ออธิบายถึงแนวทางปฏิบัติตามกฎหมายในบริบทที่กว้างขึ้น โดยให้ครอบคลุมการปฏิบัติตามกฎหมายและการป้องกันการละเมิดกฎหมาย:
                    - เพิ่มรายละเอียดของขั้นตอนการปฏิบัติตามกฎหมายในแต่ละขั้นตอน ตั้งแต่การประเมินสถานการณ์ไปจนถึงการดำเนินการตามข้อกำหนดทางกฎหมาย
                    - อธิบายวิธีการติดต่อหรือขั้นตอนในการประสานงานกับหน่วยงานทางกฎหมายที่เกี่ยวข้อง เช่น กระบวนการแจ้งความ การยื่นฟ้อง การป้องกันตนเอง หรือการรายงานเหตุการณ์
                    - ระบุถึงกฎหมายและมาตราที่เกี่ยวข้องกับสถานการณ์นี้ พร้อมทั้งให้รายละเอียดถึงแนวทางที่กฎหมายแนะนำว่าควรปฏิบัติอย่างไรในสถานการณ์นี้
                    - เพิ่มคำแนะนำในการป้องกันการกระทำที่อาจเข้าข่ายผิดกฎหมาย เพื่อให้บุคคลสามารถปฏิบัติตามกฎหมายได้อย่างถูกต้องและปลอดภัย

                    คำถาม: '{question}'
                    """
            if context_content:
                prompt += f"\n\nข้อมูลอ้างอิงที่เกี่ยวข้อง:\n{context_content}"

    elif purpose == "answer":
        # Prompt สำหรับการตอบคำถาม
        if question_type == 0:
            prompt = f"""
                    คุณเป็นทนายความผู้เชี่ยวชาญในกฎหมาย โปรดตอบคำถามเกี่ยวกับมาตราและความผิด โดยให้ข้อมูลที่ชัดเจนและครอบคลุมตามเอกสารที่เกี่ยวข้อง
                     โปรดตอบคำถามเกี่ยวกับมาตราและความผิด โดยระบุเฉพาะบทกฎหมายที่เกี่ยวข้อง และเน้นตอบคำถามโดยตรงโดยไม่ต้องกล่าวนำหรือเพิ่มข้อความขอบคุณ:
                    - ระบุบทกฎหมายที่เหมาะสมและบทลงโทษตามกฎหมายที่เกี่ยวข้อง
                    - ใช้เนื้อหาจากเอกสารอ้างอิงเพื่ออธิบายข้อหาที่เกี่ยวข้อง
                    คำถาม: '{question[:max_token_count]}'
                    """
            if context_content:
                prompt += f"\n\nข้อมูลอ้างอิงที่เกี่ยวข้อง:\n{context_content}"

        elif question_type == 1:
            prompt = f"""
                    คุณเป็นทนายความที่เชี่ยวชาญด้านการตีความกฎหมาย โปรดให้คำตอบเกี่ยวกับแนวคิดหรือคำจำกัดความทางกฎหมาย โดยอ้างอิงจากเอกสารที่เกี่ยวข้อง
                    โปรดตอบคำถามเกี่ยวกับมาตราและความผิด โดยระบุเฉพาะบทกฎหมายที่เกี่ยวข้อง และเน้นตอบคำถามโดยตรงโดยไม่ต้องกล่าวนำหรือเพิ่มข้อความขอบคุณ:
                    - อธิบายแนวคิดหรือคำจำกัดความที่เกี่ยวข้อง
                    - ใช้เนื้อหาอ้างอิงเพื่อความแม่นยำ
                    คำถาม: '{question[:max_token_count]}'
                    """
            if context_content:
                prompt += f"\n\nข้อมูลอ้างอิงที่เกี่ยวข้อง:\n{context_content}"

        elif question_type == 2:
            prompt = f"""ฟ
                    คุณเป็นทนายความที่มีความเชี่ยวชาญในการปฏิบัติตามกฎหมาย โปรดให้แนวทางปฏิบัติที่ชัดเจนในการปฏิบัติตามกฎหมาย
                    โปรดตอบคำถามเกี่ยวกับมาตราและความผิด โดยระบุเฉพาะบทกฎหมายที่เกี่ยวข้อง และเน้นตอบคำถามโดยตรงโดยไม่ต้องกล่าวนำหรือเพิ่มข้อความขอบคุณ:
                    - อธิบายขั้นตอนการดำเนินการ เช่น การแจ้งความกลับหรือการยื่นเรื่องกับหน่วยงานที่เกี่ยวข้อง
                    - อ้างอิงข้อมูลจากเอกสารที่เกี่ยวข้องเพื่อสนับสนุนคำแนะนำ
                    คำถาม: '{question[:max_token_count]}'
                    """
            if context_content:
                prompt += f"\n\nข้อมูลอ้างอิงที่เกี่ยวข้อง:\n{context_content}"

    return prompt


def calculate_similarity(question_embedding, docs):
    results = []
    for doc in docs:
        doc_embedding = doc.get("embedding")
        if doc_embedding:
            similarity = cosine_similarity(question_embedding, doc_embedding)
            if similarity >= 0.95:  # กรองเฉพาะที่มี similarity สูงกว่า threshold
                results.append((doc, similarity))
    return sorted(results, key=lambda x: x[1], reverse=True)


def question_type_matches_doc_type(doc, question_type):
    """
    Return True if the document type aligns with the question type,
    ensuring only relevant documents are included in the context.
    """
    doc_type = doc.get("ประเภทกฎหมาย")
    if question_type == 0 and doc_type == "มาตราและความผิด":
        return True
    elif question_type == 1 and doc_type == "ความหมายและการตีความ":
        return True
    elif question_type == 2 and doc_type == "แนวทางปฏิบัติ":
        return True
    return False


# ฟังก์ชันสำหรับตรวจสอบเงื่อนไขการขยายคำถาม
def should_expand_question(question):
    if not question.strip():
        # st.write("เงื่อนไข: ไม่มีข้อความในคำถาม")  # Debug
        return False  # ถ้าไม่มีข้อความให้ส่งกลับ False

    keywords = ["คืออะไร", "อย่างไร", "ทำไม", "เพราะอะไร", "เมื่อไหร่", "ต้องทำอะไร", "แบบไหน", "เพื่ออะไร", "ทำยังไง",
                "ผิดอะไรมั้ย", "เป็นอะไรมั้ย", "หรือไม่"]
    words = word_tokenize(question, keep_whitespace=False)  # ตัดคำแบบไม่มีช่องว่าง
    # st.write(f"คำที่ตัดมาได้: {words}")  # Debug

    if any(keyword in question for keyword in keywords):
        # st.write("เงื่อนไข: พบคีย์เวิร์ดที่ตรงกับคำถาม ไม่ต้องขยาย")  # Debug
        return False  # ถ้าพบคีย์เวิร์ด ไม่ต้องขยายคำถาม

    # st.write("เงื่อนไข: คำถามนี้ควรขยาย")  # Debug
    return True


# ฟังก์ชันหลักที่รวมการขยายคำถาม การค้นหาเอกสาร และการตอบคำถาม
def ask_question_to_claude(question):
    start_time = time.time()

    corrected_question = question
    question_type = classify_question_type(corrected_question)

    if question_type == 3:
        return "นี้ไม่ใช่คำถามเกี่ยวกับกฎหมาย", [], None

    should_expand = should_expand_question(corrected_question)

    llm = get_llm()
    # ขยายคำถามหากจำเป็น
    if should_expand:
        prompt_for_expansion = generate_prompt(corrected_question, question_type, purpose="expand")
        expanded = llm(prompt_for_expansion, max_length=512)[0]["generated_text"].strip()
    else:
        expanded = corrected_question

    # ไม่มี Bedrock แล้ว ใช้ embedding แบบอื่นแทน (ตรงนี้ยังต้องปรับตามโมเดล embedding ที่ใช้ใหม่)
    embedder = get_embedder()
    question_embedding = embedder.encode(expanded).tolist()
    # ถ้าเลิกใช้ Bedrock แล้วต้องเปลี่ยนด้วย

    top_docs = get_top_documents_with_similarity(question_embedding, threshold=0.95, top_k=10)

    answer_prompt = generate_prompt(expanded, question_type, purpose="answer", top_docs=top_docs)
    response = llm(answer_prompt, max_length=1024)[0]["generated_text"].strip()

    if not response:
        return "ไม่มีคำตอบ", top_docs, answer_prompt

    if response.startswith("ขอบคุณสำหรับคำถาม"):
        response = response.split("\n", 1)[-1].strip()

    end_time = time.time()
    st.write(f"เวลาที่ใช้ในการประมวลผล: {end_time - start_time:.2f} วินาที")

    return response, top_docs, answer_prompt


# # ฟังก์ชันนี้ค้นหาเอกสารที่มีค่า Cosine Similarity สูงสุดกับ embedding ของคำถามที่ขยายแล้ว โดยกรองให้ได้เฉพาะเอกสารที่มีสถานะ "Activate" และเรียงลำดับตามความคล้ายมากที่สุด
# def get_top_documents_with_similarity(question_embedding, threshold=0.95, top_k=5):
#     """
#     Find documents with cosine similarity <= threshold * top 1 similarity (filtered by top_k).
#     ค้นหาเอกสารที่มี status เป็น 'Activate' เท่านั้น
#     """
#     # Fetch documents with 'Activate' status only once to reduce database calls
#     docs = list(document_collection.find({"status": "Activate"}))
#
#     docs_with_scores = []
#     for doc in docs:
#         doc_embedding = doc.get("embedding")
#         if doc_embedding:
#             similarity = cosine_similarity(question_embedding, doc_embedding)
#             docs_with_scores.append((doc, similarity))  # Store both document and score as a tuple
#
#     # Sort documents by similarity score (highest to lowest)
#     sorted_docs = sorted(docs_with_scores, key=lambda x: x[1], reverse=True)
#
#     if not sorted_docs:
#         return []
#
#     # Get the similarity score of the top 1 document
#     top_similarity = sorted_docs[0][1]
#     top_doc = sorted_docs[0][0]  # Get the document with the highest similarity
#
#     # Calculate the 95% threshold of the top similarity
#     similarity_threshold = threshold * top_similarity
#
#     # Display Top 1 document and its similarity
#     # st.write("Top 1 Document Similarity:")
#     # st.write(f"Document ID: {top_doc['_id']}, Similarity: {top_similarity}")
#     # st.write(f"95% of Top 1 Similarity: {similarity_threshold}")
#
#     # Filter documents with similarity >= 95% of the top 1 similarity
#     filtered_docs = [(doc, score) for doc, score in sorted_docs if score >= similarity_threshold]
#
#     # Display documents that meet the similarity criteria
#     # st.write(f"Documents with Similarity >= {threshold * 100}% of Top 1:")
#     # for doc, score in filtered_docs[:top_k]:
#     #     st.write(f"Document ID: {doc['_id']}, Similarity: {score}")
#     # st.write(filtered_docs[:top_k]) # แสดง top 1
#     return filtered_docs[:top_k]  # Limit to top_k documents


# ฟังก์ชั่นดึง doc ทั้งหมดที่เกี่ยวข้อง เช่น โมเดลค้นพบเอกสารที่ตรงกับคำถามที่ chunk_index เป็น 4 (จากทั้งหมด 6 chunk) โค้ดที่เราเขียนไว้จะดึงเอา chunk ทั้งหมดของเอกสารนั้น (ทั้งหมด 6 chunk) มาเรียงตามลำดับ chunk_index และส่งกลับไปที่โมเดลเพื่อสร้างคำตอบ

def get_top_documents_with_similarity(question_embedding, threshold=0.85, top_k=10):
    docs = list(
        document_collection.find({"status": "Activate"}, {"chunk_content": 1, "embedding": 1, "document_id": 1}))
    docs_with_scores = []

    for doc in docs:
        doc_embedding = doc.get("embedding")
        if doc_embedding:
            similarity = cosine_similarity(question_embedding, doc_embedding)
            docs_with_scores.append((doc, similarity))  # Store both document and score as a tuple

    # Sort documents by similarity score (highest to lowest)
    sorted_docs = sorted(docs_with_scores, key=lambda x: x[1], reverse=True)

    # ลด threshold เพื่อให้คืนเอกสารที่คล้ายคลึงมากขึ้น
    top_similarity = sorted_docs[0][1] if sorted_docs else 0
    similarity_threshold = threshold * top_similarity
    filtered_docs = [(doc, score) for doc, score in sorted_docs if score >= similarity_threshold]

    return filtered_docs[:top_k]  # Return only the top_k documents

    # Display Top 1 document and its similarity
    # st.write("Top 1 Document Similarity:")
    # st.write(f"Document ID: {top_doc['_id']}, Similarity: {top_similarity}")
    # st.write(f"95% of Top 1 Similarity: {similarity_threshold}")

    # Display documents that meet the similarity criteria
    # st.write(f"Documents with Similarity >= {threshold * 100}% of Top 1:")
    # for doc, score in filtered_docs[:top_k]:
    #     st.write(f"Document ID: {doc['_id']}, Similarity: {score}")
    # st.write("Filtered documents:", filtered_docs[:top_k])

    # Collect all chunks of each unique document_id in filtered_docs
    unique_docs = {}
    for doc, score in filtered_docs:
        document_id = doc["document_id"]
        if document_id not in unique_docs:
            chunks = list(document_collection.find({"document_id": document_id}).sort("chunk_index", 1))
            unique_docs[document_id] = chunks
            # Debugging: Show each chunk collected
            # st.write(f"Document ID: {document_id} - Collected chunks:")
            # for chunk in chunks:
            #     st.write(f"Chunk Index: {chunk['chunk_index']}, Content: {chunk.get('chunk_content', 'N/A')}")

    # Convert dictionary to a flat list of tuples with document and similarity score
    combined_docs = [(chunk, cosine_similarity(question_embedding, chunk["embedding"])) for doc_id, chunks in
                     unique_docs.items() for chunk in chunks]

    return combined_docs[:top_k]


# ฟังก์ชั่นดึง doc ทั้งหมดที่เกี่ยวข้อง เช่น โมเดลค้นพบเอกสารที่ตรงกับคำถามที่ chunk_index เป็น 4 (จากทั้งหมด 6 chunk) โค้ดที่เราเขียนไว้จะดึงเอา chunk ทั้งหมดของเอกสารนั้น (ทั้งหมด 6 chunk) มาเรียงตามลำดับ chunk_index และส่งกลับไปที่โมเดลเพื่อสร้างคำตอบ


# ฟังก์ชันนี้ช่วยสร้าง prompt ที่มีทั้งคำถามที่ขยายแล้วและบริบทที่เกี่ยวข้องจากเอกสาร ทำให้โมเดล LLM สามารถให้คำตอบที่มีความแม่นยำและสอดคล้องกับข้อมูลที่ดึงมา
def create_prompt_with_expanded_question_and_context(expanded_question, top_docs, question_type):
    """
    Create a prompt that combines the expanded question from Claude and the context from MongoDB,
    filtered and structured according to the question type.
    """
    context_parts = []

    # Filter and structure context based on question_type relevance
    for doc, score in top_docs:
        if question_type_matches_doc_type(doc, question_type):  # Implement a function to match types
            law_type = doc.get("ประเภทกฎหมาย", "N/A")
            section = doc.get("ภาค", "N/A")
            sub_section = doc.get("ลักษณะ", "N/A")
            category = doc.get("หมวด", "N/A")
            section_number = doc.get("มาตรา", "N/A")
            content = doc.get("chunk_content", "")

            # Create a structured context part for each document
            context_part = f"ประเภท: {law_type}, ภาค: {section}, ลักษณะ: {sub_section}, หมวด: {category}, มาตรา: {section_number}\nเนื้อหา: {content}"
            context_parts.append(context_part)

    # Combine all context parts into one string
    combined_context = "\n\n".join(context_parts)

    # Create the final prompt with the expanded question and the combined context
    # Final prompt combining expanded question with filtered context
    prompt = f"""
       คำถามจากผู้ใช้: {expanded_question}

       เนื้อหาที่เกี่ยวข้องสำหรับการตอบคำถาม:
       {combined_context}

       โปรดตอบคำถามดังกล่าวโดยอ้างอิงเฉพาะข้อมูลที่ปรากฏใน "เนื้อหาที่เกี่ยวข้องสำหรับการตอบคำถาม" ด้านบนนี้เท่านั้น และปฏิบัติตามคำแนะนำดังต่อไปนี้:
       1. วิเคราะห์คำถามจากผู้ใช้อย่างรอบคอบ และใช้ข้อมูลในเอกสารที่เกี่ยวข้องเพื่อให้คำตอบที่ชัดเจนและครอบคลุม
       2. ห้ามคัดลอกเนื้อหาในเอกสารโดยตรง ให้สรุปข้อมูลและเชื่อมโยงกับคำถามอย่างเหมาะสม
       3. หากข้อมูลในเอกสารไม่เพียงพอที่จะตอบคำถามได้อย่างครบถ้วน ให้ระบุว่า "ไม่มีข้อมูลที่เกี่ยวข้อง" โดยไม่คาดเดาคำตอบหรือเพิ่มข้อมูลที่ไม่ปรากฏในเอกสาร
       4. ใช้ภาษาที่เป็นทางการ ชัดเจน และกระชับ

       หมายเหตุ:
       - หากคำถามเกี่ยวข้องกับมาตรากฎหมาย ให้ระบุชื่อกฎหมาย มาตรา และเนื้อหาที่เกี่ยวข้องอย่างชัดเจน
       - หากคำถามเกี่ยวกับแนวทางปฏิบัติ ให้สรุปขั้นตอนและข้อปฏิบัติตามเนื้อหาที่ระบุไว้ในเอกสาร

       ตัวอย่างการตอบ:
       - กรณีที่ข้อมูลเพียงพอ: "มาตรา 5 ของกฎหมายว่าด้วย... ระบุว่า..."
       - กรณีไม่มีข้อมูลที่เกี่ยวข้อง: "ไม่มีข้อมูลที่เกี่ยวข้อง"
       """
    return prompt

    # Print or return the generated prompt with the expanded question and context
    # st.write("Generated Prompt with Expanded Question and Context:")
    # st.write(prompt)

    # return prompt


# Save chat to MongoDB
def save_chat_to_db(username, question, response, context):
    chat_history = {
        "username": username,
        "question": question,
        "response": response,
        "context": context,
        "timestamp": datetime.utcnow()
    }
    try:
        conversation_collection.insert_one(chat_history)
    except Exception as e:
        st.error(f"Error saving chat: {str(e)}")


# สิ้นสุดการถามทั้งหมด------------------------------------------------------------------------------------------------------------------------------------------สิ้นสุดการถามทั้งหมด

# ฟังก์ชันแบ่งข้อความเป็นส่วนๆ
def split_text(text, max_size=2048):
    """แบ่งข้อความออกเป็นหลายส่วน โดยแต่ละส่วนไม่เกิน max_size"""
    chunks = []
    while len(text) > max_size:
        # หาจุดตัดที่ใกล้กับขอบเขต max_size มากที่สุด โดยตัดที่ช่องว่าง
        split_index = text[:max_size].rfind(' ')
        if split_index == -1:  # ถ้าไม่มีช่องว่างให้ตัดตรง max_size
            split_index = max_size
        chunks.append(text[:split_index].strip())
        text = text[split_index:].strip()
    chunks.append(text)  # เพิ่มส่วนสุดท้ายที่เหลือ
    return chunks


# ฟังก์ชันแยกสำหรับการส่งคำถามไปยัง LLM โดยตรงเมื่อ similarity ต่ำกว่าเกณฑ์
def ask_question_to_llm(question):
    """
    Function to ask question to LLM directly without context from MongoDB documents.
    """
    llm = get_llm()
    if not llm:
        st.write("Error initializing the LLM.")
        return "Error initializing the LLM."

    # แสดง prompt ที่จะส่งไปยัง LLM
    st.write("Prompt ที่ส่งไปยัง LLM (กรณี similarity ต่ำกว่า 0.6):")
    st.write(question)

    try:
        # ส่งคำถามไปที่ LLM โดยตรง
        response = llm(question, max_length=1024)[0]['generated_text'].strip()
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดจาก LLM: {str(e)}")
        return "ไม่สามารถเรียกใช้โมเดลภาษาได้ในขณะนี้"

    return response



# ฟังก์ชัน update_document_status ใช้สำหรับ อัปเดตสถานะของเอกสารในฐานข้อมูล โดยจะทำการค้นหาเอกสารที่มีประเภทและลักษณะเดียวกันและ เปลี่ยนสถานะเอกสารเก่าเป็น "Deactivate" คงเหลือไว้เพียงเอกสารใหม่ที่สุดในกลุ่มเดียวกันเท่านั้น
def update_document_status(law_type, section, sub_section, category, section_number, draft_date):
    """
    Update document statuses: Deactivate older documents and Activate the newest one.
    """
    # Step 1: Find documents with the same law_type, section, sub_section, category, and section_number
    matching_docs = list(document_collection.find({
        "ประเภทกฎหมาย": law_type,
        "ภาค": section,
        "ลักษณะ": sub_section,
        "หมวด": category,
        "มาตรา": section_number
    }))

    # ตรวจสอบว่ามีเอกสารถูกค้นพบหรือไม่
    print(f"Found {len(matching_docs)} matching documents.")

    # Step 2: Check if there are any matching documents and deactivate the older ones
    for doc in matching_docs:
        print(f"Checking document: {doc['_id']}, Draft Date: {doc.get('วันที่ร่าง')}")

        # Convert the draft date from the document to a datetime object for comparison
        existing_draft_date = datetime.fromisoformat(doc.get("วันที่ร่าง"))

        # If existing_draft_date is datetime, convert it to date for comparison
        if isinstance(existing_draft_date, datetime):
            existing_draft_date_only = existing_draft_date.date()
        else:
            existing_draft_date_only = existing_draft_date

        # If draft_date is datetime, convert it to date for comparison
        if isinstance(draft_date, datetime):
            new_draft_date_only = draft_date.date()
        else:
            new_draft_date_only = draft_date

        print(f"Existing Draft Date: {existing_draft_date_only}, New Draft Date: {new_draft_date_only}")

        # If the existing document's draft date is older than the new draft date, deactivate it
        if existing_draft_date_only < new_draft_date_only:
            print(f"Deactivating document {doc['_id']}")
            document_collection.update_one(
                {"_id": doc["_id"]},
                {"$set": {"status": "Deactivate"}}
            )
        else:
            print(f"Document {doc['_id']} is newer or same, not deactivating.")

    # Step 3: Return true if we deactivate the older documents
    return True


# ฟังก์ชัน toggle_document_status ใช้สำหรับ สลับสถานะของเอกสาร ระหว่าง "Activate" และ "Deactivate" ขึ้นอยู่กับสถานะปัจจุบันในฐานข้อมูล MongoDB
def toggle_document_status(document_id):
    """
    Toggle document status between Activate and Deactivate based on current status.
    """
    # Get the current status of the document
    doc = document_collection.find_one({"_id": ObjectId(document_id)})

    if doc:
        new_status = "Activate" if doc["status"] == "Deactivate" else "Deactivate"

        # Update the status in MongoDB
        document_collection.update_one({"_id": ObjectId(document_id)}, {"$set": {"status": new_status}})
        st.success(f"Document {document_id} status changed to {new_status}.")
    else:
        st.error("Document not found.")


# ฟังก์ชัน store_law_data_in_mongo ใช้สำหรับ เก็บข้อมูลกฎหมายใหม่ ลงใน MongoDB โดยทำให้เอกสารใหม่นั้นมีสถานะ "Activate" และ ปิดใช้งานเอกสารเก่าที่มีประเภทและรายละเอียดเดียวกัน (เช่น ประเภทกฎหมาย, ภาค, ลักษณะ, หมวด, และ มาตรา)
def store_law_data_in_mongo(law_type, draft_date, section_number, law_content, section, sub_section=None,
                            category=None):
    """
    Store new law data in MongoDB and ensure that the new document becomes Active while deactivating older ones.
    """
    # Generate a unique document ID for the new document
    document_id = str(uuid4())

    # Clean and tokenize the content
    cleaned_content = clean_and_tokenize(law_content)

    # Split the content into chunks
    chunks = split_text(cleaned_content, max_size=2048)

    # Embed the document chunks
    embedder = get_embedder()
    embeddings = embedder.encode(chunks).tolist()


    # Step 1: Deactivate older documents with the same law_type, section, sub_section, category, and section_number
    update_document_status(law_type, section, sub_section, category, section_number, draft_date)

    # Step 2: Store the new document as Active
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        chunk_data = {
            "document_id": document_id,
            "chunk_index": i,
            "ประเภทกฎหมาย": law_type,
            "วันที่ร่าง": draft_date.isoformat(),
            "ภาค": section,
            "ลักษณะ": sub_section,
            "หมวด": category,
            "มาตรา": section_number,
            "chunk_content": chunk,
            "embedding": embedding,
            "status": "Activate",  # Mark this new document as Active
            "timestamp": datetime.utcnow()
        }
        try:
            # Insert the new document into MongoDB
            document_collection.insert_one(chunk_data)
        except Exception as e:
            st.error(f"Error storing document in MongoDB: {str(e)}")


# ฟังก์ชันสำหรับดึงข้อมูลเอกสารที่เลือกจาก MongoDB
def get_document_by_id(document_id):
    try:
        # ดึงเอกสารจาก MongoDB ตาม ID
        document = document_collection.find_one({"_id": ObjectId(document_id)})
        return document
    except Exception as e:
        st.error(f"Error fetching document: {str(e)}")
        return None


# ฟังก์ชันสำหรับดึงข้อมูลเอกสารจาก MongoDB เพื่อแสดงใน dropdown menu
def get_document_list():
    try:
        # ดึงเอกสารทั้งหมดจาก MongoDB (คุณสามารถปรับฟิลด์ที่ต้องการดึงได้)
        documents = document_collection.find({}, {"_id": 1, "ประเภทกฎหมาย": 1, "ภาค": 1, "มาตรา": 1})

        # Return เป็น tuple (document_id, ชื่อเอกสารที่จะแสดงใน dropdown)
        return [(str(doc["_id"]),
                 f"ประเภท: {doc.get('ประเภทกฎหมาย', 'N/A')}, ภาค: {doc.get('ภาค', 'N/A')}, มาตรา: {doc.get('มาตรา', 'N/A')}")
                for doc in documents]
    except Exception as e:
        st.error(f"Error fetching documents: {str(e)}")
        return []


# ฟังก์ชันสำหรับรีเฟรชแอปด้วยการตั้งค่า query parameters ใหม่
def rerun_app():
    st.experimental_set_query_params(updated=str(st.session_state.get("updated", 0)))


# ปรับปรุงในฟังก์ชัน main
def main():
    st.title("Make The Law")

    add_css()
    tab1, tab2, tab3 = st.tabs(["Ask a Question", "Upload Law Data", "Change Status"])

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # New Chat reset
    if st.sidebar.button("New Chat"):
        st.session_state["selected_chat"] = None
        st.session_state["messages"] = []

    show_chat_history_in_sidebar("User")

    # ช่องใส่คำถามอยู่ด้านล่างของหน้าจอในทุกแท็บ
    user_input_question = st.chat_input("Enter your question:")

    with tab1:
        for message in st.session_state["messages"]:
            role = message["role"]
            content = message["content"]
            with st.chat_message(role):
                st.markdown(content)

        if user_input_question:
            # แสดงคำถามของผู้ใช้และบันทึกใน session state
            st.session_state.messages.append({"role": "user", "content": user_input_question})
            with st.chat_message("user"):
                st.markdown(user_input_question)

            # AI typing animation
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                # เริ่มฟังก์ชัน animation ใน background thread
                with st.spinner("AI is Thinking..."):
                    ai_response, top_docs, prompt_n = ask_question_to_claude(user_input_question)
                # ถาม AI ด้วยคำถามที่เชื่อมโยงกัน

                # แสดงผลลัพธ์จาก AI
                if isinstance(ai_response, dict):
                    message_placeholder.markdown("นี่คือข้อมูลของคุณ")
                    message_placeholder.table(pd.DataFrame.from_dict(ai_response))
                else:
                    display_ai_message_gradually(ai_response, message_placeholder)

            # บันทึกคำตอบของ AI
            st.session_state.messages.append({"role": "assistant", "content": ai_response})

            # บันทึกลงใน MongoDB
            save_chat_to_db("User", user_input_question, ai_response,
                            prompt_n)

    with tab2:
        st.subheader("Upload Law Data")
        st.subheader("Manage Document Status")
        law_types = ["กรุณาเลือกประเภทของกฏหมาย", "อาญา", "แพ่ง", "พรบ คอมพิวเตอร์", "PDPA", "เคสตัวอย่าง",
                     "พรบ การจัดซื้อจัดจ้าง", "พรบ ส่งเสริมและรักษาสิ่งแวดล้อม"]
        selected_law_type = st.selectbox("ประเภทของกฏหมาย", law_types, key="law_type")

        if selected_law_type == "อาญา":
            draft_date = st.date_input("วันที่ร่าง", datetime.now(), key="draft_date")
            sections = {
                "ภาค ๑ บทบัญญัติทั่วไป (มาตรา ๑ - ๑๐๖)": {
                    "ลักษณะ ๑ บทบัญญัติที่ใช้แก่ความผิดทั่วไป (มาตรา ๑ - ๑๐๑)": {
                        "หมวด ๑ บทนิยาม (มาตรา ๑)": (1, 1),
                        "หมวด ๒ การใช้กฎหมายอาญา (มาตรา ๒ - ๑๗)": (2, 17),
                        "หมวด ๓ โทษและวิธีการเพื่อความปลอดภัย (มาตรา ๑๘ - ๕๘)": (18, 58),
                        "หมวด ๔ ความรับผิดในทางอาญา (มาตรา ๕๙ - ๗๙)": (59, 79),
                        "หมวด ๕ การพยายามกระทำความผิด (มาตรา ๘๐ - ๘๒)": (80, 82),
                        "หมวด ๖ ตัวการและผู้สนับสนุน (มาตรา ๘๓ - ๘๙)": (83, 89),
                        "หมวด ๗ การกระทำความผิดหลายบทหรือหลายกระทง (มาตรา ๙๐ - ๙๑)": (90, 91),
                        "หมวด ๘ การกระทำความผิดอีก (มาตรา ๙๒ - ๙๔)": (92, 94),
                        "หมวด ๙ อายุความ (มาตรา ๙๕ - ๑๐๑)": (95, 101)
                    },
                    "ลักษณะ ๒ บทบัญญัติที่ใช้แก่ความผิดลหุโทษ (มาตรา ๑๐๒ - ๑๐๖)": (102, 106)
                },
                "ภาค ๒ ความผิด (มาตรา ๑๐๗ - ๓๖๖/๔)": {
                    "ลักษณะ ๑ ความผิดเกี่ยวกับความมั่นคงแห่งราชอาณาจักร (มาตรา ๑๐๗ - ๑๓๕)": {
                        "หมวด ๑ ความผิดต่อองค์พระมหากษัตริย์ พระราชินี รัชทายาท และผู้สำเร็จราชการแทนพระองค์ (มาตรา ๑๐๗ - ๑๑๒)": (
                            107, 112),
                        "หมวด ๒ ความผิดต่อความมั่นคงของรัฐภายในราชอาณาจักร (มาตรา ๑๑๓ - ๑๑๘)": (113, 118),
                        "หมวด ๓ ความผิดต่อความมั่นคงของรัฐภายนอกราชอาณาจักร (มาตรา ๑๑๙ - ๑๒๙)": (119, 129),
                        "หมวด ๔ ความผิดต่อสัมพันธไมตรีกับต่างประเทศ (มาตรา ๑๓๐ - ๑๓๕)": (130, 135)
                    },
                    "ลักษณะ ๑/๑ ความผิดเกี่ยวกับการก่อการร้าย (มาตรา ๑๓๕/๑ - ๑๓๕/๔)": (135.1, 135.4),
                    "ลักษณะ ๒ ความผิดเกี่ยวกับการปกครอง (มาตรา ๑๓๖ - ๑๖๖)": {
                        "หมวด ๑ ความผิดต่อเจ้าพนักงาน (มาตรา ๑๓๖ - ๑๔๖)": (136, 146),
                        "หมวด ๒ ความผิดต่อตำแหน่งหน้าที่ราชการ (มาตรา ๑๔๗ - ๑๖๖)": (147, 166)
                    },
                    "ลักษณะ ๓ ความผิดเกี่ยวกับการยุติธรรม (มาตรา ๑๖๗ - ๒๐๕)": {
                        "หมวด ๑ ความผิดต่อเจ้าพนักงานในการยุติธรรม (มาตรา ๑๖๗ - ๑๙๙)": (167, 199),
                        "หมวด ๒ ความผิดต่อตำแหน่งหน้าที่ในการยุติธรรม (มาตรา ๒๐๐ - ๒๐๕)": (200, 205)
                    },
                    "ลักษณะ ๔ ความผิดเกี่ยวกับศาสนา (มาตรา ๒๐๖ - ๒๐๘)": (206, 208),
                    "ลักษณะ ๕ ความผิดเกี่ยวกับความสงบสุขของประชาชน (มาตรา ๒๐๙ - ๒๑๖)": (209, 216),
                    "ลักษณะ ๖ ความผิดเกี่ยวกับการก่อให้เกิดภยันตรายต่อประชาชน (มาตรา ๒๑๗ - ๒๓๙)": (217, 239),
                    "ลักษณะ ๗ ความผิดเกี่ยวกับการปลอมและการแปลง (มาตรา ๒๔๐ - ๒๖๙/๑๕)": {
                        "หมวด ๑ ความผิดเกี่ยวกับเงินตรา (มาตรา ๒๔๐ - ๒๔๙)": (240, 249),
                        "หมวด ๒ ความผิดเกี่ยวกับดวงตรา แสตมป์และตั๋ว (มาตรา ๒๕๐ - ๒๖๓)": (250, 263),
                        "หมวด ๓ ความผิดเกี่ยวกับเอกสาร (มาตรา ๒๖๔ - ๒๖๙)": (264, 269),
                        "หมวด ๔ ความผิดเกี่ยวกับบัตรอิเล็กทรอนิกส์ (มาตรา ๒๖๙/๑ - ๒๖๙/๗)": (269.1, 269.7),
                        "หมวด ๕ ความผิดเกี่ยวกับหนังสือเดินทาง (มาตรา ๒๖๙/๘ - ๒๖๙/๑๕)": (269.10, 269.9)
                    },
                    "ลักษณะ ๘ ความผิดเกี่ยวกับการค้า (มาตรา ๒๗๐ - ๒๗๕)": (270, 275),
                    "ลักษณะ ๙ ความผิดเกี่ยวกับเพศ (มาตรา ๒๗๖ - ๒๘๗/๒)": (276, 287.2),
                    "ลักษณะ ๑๐ ความผิดเกี่ยวกับชีวิตและร่างกาย (มาตรา ๒๘๘ - ๓๐๘)": {
                        "หมวด ๑ ความผิดต่อชีวิต (มาตรา ๒๘๘ - ๒๙๔)": (288, 294),
                        "หมวด ๒ ความผิดต่อร่างกาย (มาตรา ๒๙๕ - ๓๐๐)": (295, 300),
                        "หมวด ๓ ความผิดฐานทำให้แท้งลูก (มาตรา ๓๐๑ - ๓๐๕)": (301, 305),
                        "หมวด ๔ ความผิดฐานทอดทิ้งเด็ก คนป่วยเจ็บหรือคนชรา (มาตรา ๓๐๖ - ๓๐๘)": (306, 308)
                    },
                    "ลักษณะ ๑๑ ความผิดเกี่ยวกับเสรีภาพและชื่อเสียง (มาตรา ๓๐๙ - ๓๓๓)": {
                        "หมวด ๑ ความผิดต่อเสรีภาพ (มาตรา ๓๐๙ - ๓๒๑/๑)": (309, 321.1),
                        "หมวด ๒ ความผิดฐานเปิดเผยความลับ (มาตรา ๓๒๒ - ๓๒๕)": (322, 325),
                        "หมวด ๓ ความผิดฐานหมิ่นประมาท (มาตรา ๓๒๖ - ๓๓๓)": (326, 333)
                    },
                    "ลักษณะ ๑๒ ความผิดเกี่ยวกับทรัพย์ (มาตรา ๓๓๔ - ๓๖๖)": {
                        "หมวด ๑ ความผิดฐานลักทรัพย์และวิ่งราวทรัพย์ (มาตรา ๓๓๔ - ๓๓๖ ทวิ)": (334, 336.2),
                        "หมวด ๒ ความผิดฐานกรรโชก รีดเอาทรัพย์ ชิงทรัพย์และปล้นทรัพย์ (มาตรา ๓๓๗ - ๓๔๐ ตรี)": (
                            337, 340.3),
                        "หมวด ๓ ความผิดฐานฉ้อโกง (มาตรา ๓๔๑ - ๓๔๘)": (341, 348),
                        "หมวด ๔ ความผิดฐานโกงเจ้าหนี้ (มาตรา ๓๔๙ - ๓๕๑)": (349, 351),
                        "หมวด ๕ ความผิดฐานยักยอก (มาตรา ๓๕๒ - ๓๕๖)": (352, 356),
                        "หมวด ๖ ความผิดฐานรับของโจร (มาตรา ๓๕๗)": (357, 357),
                        "หมวด ๗ ความผิดฐานทำให้เสียทรัพย์ (มาตรา ๓๕๘ - ๓๖๑)": (358, 361),
                        "หมวด ๘ ความผิดฐานบุกรุก (มาตรา ๓๖๒ - ๓๖๖)": (362, 366)
                    },
                    "ลักษณะ ๑๓ ความผิดเกี่ยวกับศพ (มาตรา ๓๖๖/๑ - ๓๖๖/๔)": (366.1, 366.4)
                },
                "ภาค ๓ ลหุโทษ (มาตรา ๓๖๗ - ๓๙๘)": (367, 398),
                "พระราชบัญญัติให้ใช้ประมวลกฎหมายอาญา พ.ศ. ๒๔๙๙": None,
                "เหตุผลในการประกาศใช้": None,
                "อื่นๆ": None
            }

            selected_section = st.selectbox("เลือกภาค", list(sections.keys()), key="section_unique")

            if selected_section in ["พระราชบัญญัติให้ใช้ประมวลกฎหมายอาญา พ.ศ. ๒๔๙๙", "เหตุผลในการประกาศใช้", "อื่นๆ"]:
                section_number = None  # ไม่มีมาตราในกรณีนี้
                selected_sub_section = None
                selected_category = None
            else:
                selected_sub_section = st.selectbox("เลือกลักษณะ", list(sections[selected_section]),
                                                    key="sub_section_unique")

                if isinstance(sections[selected_section][selected_sub_section], dict):
                    selected_category = st.selectbox("เลือกหมวด",
                                                     list(sections[selected_section][selected_sub_section].keys()),
                                                     key="category_unique")
                    min_section, max_section = sections[selected_section][selected_sub_section][selected_category]
                else:
                    selected_category = None
                    min_section, max_section = sections[selected_section][selected_sub_section]

                # ตรวจสอบว่าหมวดหรือลักษณะที่เลือกเป็นหมวดหรือลักษณะที่ต้องการจัดเก็บเป็นทศนิยม
                if selected_category in ["หมวด ๔ ความผิดเกี่ยวกับบัตรอิเล็กทรอนิกส์ (มาตรา ๒๖๙/๑ - ๒๖๙/๗)",
                                         "หมวด ๕ ความผิดเกี่ยวกับหนังสือเดินทาง (มาตรา ๒๖๙/๘ - ๒๖๙/๑๕)"] or \
                        selected_sub_section in ["ลักษณะ ๑/๑ ความผิดเกี่ยวกับการก่อการร้าย (มาตรา ๑๓๕/๑ - ๑๓๕/๔)",
                                                 "ลักษณะ ๙ ความผิดเกี่ยวกับเพศ (มาตรา ๒๗๖ - ๒๘๗/๒)",
                                                 "ลักษณะ ๑๓ ความผิดเกี่ยวกับศพ (มาตรา ๓๖๖/๑ - ๓๖๖/๔)"]:
                    section_number = st.number_input(
                        f"มาตรา ({min_section} - {max_section})",
                        min_value=float(min_section),
                        max_value=float(max_section),
                        step=0.01,  # ใช้ทศนิยม 2 ตำแหน่ง
                        format="%.2f",
                        key="section_number_unique"
                    )
                else:
                    section_number = st.number_input(
                        f"มาตรา ({min_section} - {max_section})",
                        min_value=int(min_section),
                        max_value=int(max_section),
                        format="%d",  # ใช้จำนวนเต็ม
                        key="section_number_unique"
                    )

            law_content = st.text_area("เนื้อหากฎหมาย", height=400, key="law_content_unique")

            if st.button("Submit", key="submit_law_btn_unique"):
                if not section_number and selected_section not in ["พระราชบัญญัติให้ใช้ประมวลกฎหมายอาญา พ.ศ. ๒๔๙๙",
                                                                   "เหตุผลในการประกาศใช้", "อื่นๆ"]:
                    st.error("กรุณากรอกข้อมูลในช่อง มาตรา")
                elif not law_content:
                    st.error("กรุณากรอกเนื้อหากฎหมาย")
                else:
                    st.success(
                        f"กฎหมายที่บันทึก:\nประเภท: {selected_law_type}\nภาค: {selected_section}\nมาตรา: {section_number or 'N/A'}\nเนื้อหากฎหมาย: {law_content}")
                    with st.spinner('กำลังฝังข้อมูล...'):
                        try:
                            store_law_data_in_mongo(selected_law_type, draft_date, section_number, law_content,
                                                    selected_section, selected_sub_section, selected_category)
                            st.success("ฝังข้อมูลสำเร็จและบันทึกลง MongoDB แล้ว!")
                        except Exception as e:
                            st.error(f"เกิดข้อผิดพลาด: {str(e)}")

        elif selected_law_type == "แพ่ง":
            draft_date = st.date_input("วันที่ร่าง", datetime.now(), key="draft_date")
            sections = {
                "พระราชบัญญัติให้ใช้ประมวลกฎหมายแพ่งและพาณิชย์": {
                    "พระราชกฤษฎีกาให้ใช้ประมวลกฎหมายแพ่งและพาณิชย์ บรรพ ๑ และ ๒ ที่ได้ตรวจชำระใหม่": None,
                    "พระราชบัญญัติให้ใช้บทบัญญัติบรรพ ๑ แห่งประมวลกฎหมายแพ่งและพาณิชย์ที่ได้ตรวจชำระใหม่ พ.ศ. ๒๕๓๕": None,
                    "พระราชกฤษฎีกาให้ใช้บทบัญญัติแห่งประมวลกฎหมายแพ่งและพาณิชย์ บรรพ ๓ ที่ได้ตรวจชำระใหม่": None,
                    "พระราชกฤษฎีกาให้ใช้บทบัญญัติ บรรพ ๔ แห่งประมวลกฎหมายแพ่งและพาณิชย์": None,
                    "พระราชบัญญัติให้ใช้บทบัญญัติบรรพ ๕ แห่งประมวลกฎหมายแพ่งและพาณิชย์ ที่ได้ตรวจชำระใหม่ พ.ศ. ๒๕๑๙": None,
                    "พระราชบัญญัติให้ใช้บทบัญญัติบรรพ ๖ แห่งประมวลกฎหมายแพ่งและพาณิชย์ พุทธศักราช ๒๔๗๗": None
                },
                "ข้อความเบื้องต้น (มาตรา ๑ - ๓)": (1, 3),
                "บรรพ ๑ หลักทั่วไป (มาตรา ๔ - ๑๙๓/๓๕)": {
                    "ลักษณะ ๑ บทเบ็ดเสร็จทั่วไป (มาตรา ๔ - ๑๔)": (4, 14),
                    "ลักษณะ ๒ บุคคล (มาตรา ๑๕ - ๑๓๖)": {
                        "หมวด ๑ บุคคลธรรมดา (มาตรา ๑๕ - ๖๔)": (15, 64),
                        "หมวด ๒ นิติบุคคล (มาตรา ๖๕ - ๑๓๖)": (65, 136)
                    },

                    "ลักษณะ ๓ ทรัพย์ (มาตรา ๑๓๗ - ๑๔๘)": (137, 148),
                    "ลักษณะ ๔ นิติกรรม (มาตรา ๑๔๙ - ๑๙๓)": {
                        "หมวด ๑ บทเบ็ดเสร็จทั่วไป (มาตรา ๑๔๙ - ๑๕๓)": (149, 153),
                        "หมวด ๒ การแสดงเจตนา (มาตรา ๑๕๔ - ๑๗๑)": (154, 171),
                        "หมวด ๓ โมฆะกรรมและโมฆียะกรรม (มาตรา ๑๗๒ - ๑๘๑)": (172, 181),
                        "หมวด ๔ เงื่อนไขและเงื่อนเวลา (มาตรา ๑๘๒ - ๑๙๓)": (182, 193)
                    },
                    "ลักษณะ ๕ ระยะเวลา (มาตรา ๑๙๓/๑ - ๑๙๓/๘)": (193.1, 193.8),
                    "ลักษณะ ๖ อายุความ (มาตรา ๑๙๓/๙ - ๑๙๓/๓๕)": {
                        "หมวด ๑ บทเบ็ดเสร็จทั่วไป (มาตรา ๑๙๓/๙ - ๑๙๓/๒๙)": (193.10, 193.9),
                        "หมวด ๒ กำหนดอายุความ (มาตรา ๑๙๓/๓๐ - ๑๙๓/๓๕)": (193.30, 193.35)
                    },
                },
                "บรรพ ๒ หนี้ (มาตรา ๑๙๔ - ๔๕๒)": {
                    "ลักษณะ ๑ บทเบ็ดเสร็จทั่วไป (มาตรา ๑๙๔ - ๓๕๓)": {
                        "หมวด ๑ วัตถุแห่งหนี้ (มาตรา ๑๙๔ - ๒๐๒)": (194, 202),
                        "หมวด ๒ ผลแห่งหนี้ (มาตรา ๒๐๓ - ๒๘๙)": (203, 289),
                        "หมวด ๓ ลูกหนี้และเจ้าหนี้หลายคน (มาตรา ๒๙๐ - ๓๐๒)": (290, 302),
                        "หมวด ๔ โอนสิทธิเรียกร้อง (มาตรา ๓๐๓ - ๓๑๓)": (303, 313),
                        "หมวด ๕ ความระงับแห่งหนี้ (มาตรา ๓๑๔ - ๓๕๓)": (314, 353)
                    },
                    "ลักษณะ ๒ สัญญา (มาตรา ๓๕๔ - ๓๙๔)": {
                        "หมวด ๑ ก่อให้เกิดสัญญา (มาตรา ๓๕๔ - ๓๖๘)": (354, 368),
                        "หมวด ๒ ผลแห่งสัญญา (มาตรา ๓๖๙ - ๓๗๖)": (369, 376),
                        "หมวด ๓ มัดจำและกำหนดเบี้ยปรับ (มาตรา ๓๗๗ - ๓๘๕)": (377, 385),
                        "หมวด ๔ เลิกสัญญา (มาตรา ๓๘๖ - ๓๙๔)": (386, 394)
                    },
                    "ลักษณะ ๓ จัดการงานนอกสั่ง (มาตรา ๓๙๕ - ๔๐๕)": (395, 405),
                    "ลักษณะ ๔ ลาภมิควรได้ (มาตรา ๔๐๖ - ๔๑๙)": (406, 419),
                    "ลักษณะ ๕ ละเมิด (มาตรา ๔๒๐ - ๔๕๒)": {
                        "หมวด ๑ ความรับผิดเพื่อละเมิด (มาตรา ๔๒๐ - ๔๓๗)": (420, 437),
                        "หมวด ๒ ค่าสินไหมทดแทนเพื่อละเมิด (มาตรา ๔๓๘ - ๔๔๘)": (438, 448),
                        "หมวด ๓ นิรโทษกรรม (มาตรา ๔๔๙ - ๔๕๒)": (449, 452)
                    },
                },
                "บรรพ ๓ เอกเทศสัญญา (มาตรา ๔๕๓ - ๑๒๙๗)": {
                    "ลักษณะ ๑ ซื้อขาย (มาตรา ๔๕๓ - ๕๑๗)": {
                        "หมวด ๑ สภาพและหลักสำคัญของสัญญาซื้อขาย (มาตรา ๔๕๓ - ๔๖๐)": (453, 460),
                        "หมวด ๒ หน้าที่และความรับผิดของผู้ขาย (มาตรา ๔๖๑ - ๔๘๕)": (461, 485),
                        "หมวด ๓ หน้าที่ของผู้ซื้อ (มาตรา ๔๘๖ - ๔๙๐)": (486, 490),
                        "หมวด ๔ การซื้อขายเฉพาะบางอย่าง (มาตรา ๔๙๑ - ๕๑๗)": (491, 517)
                    },
                    "ลักษณะ ๒ แลกเปลี่ยน (มาตรา ๕๑๘ - ๕๒๐)": (518, 520),
                    "ลักษณะ ๓ ให้ (มาตรา ๕๒๑ - ๕๓๖)": (521, 536),
                    "ลักษณะ ๔ เช่าทรัพย์ (มาตรา ๕๓๗ - ๕๗๑)": {
                        "หมวด ๑ บทเบ็ดเสร็จทั่วไป (มาตรา ๕๓๗ - ๕๔๕)": (537, 545),
                        "หมวด ๒ หน้าที่และความรับผิดของผู้ให้เช่า (มาตรา ๕๔๖ - ๕๕๑)": (546, 551),
                        "หมวด ๓ หน้าที่และความรับผิดของผู้เช่า (มาตรา ๕๕๒ - ๕๖๓)": (552, 563),
                        "หมวด ๔ ความระงับแห่งสัญญาเช่า (มาตรา ๕๖๔ - ๕๗๑)": (564, 571)
                    },
                    "ลักษณะ ๕ เช่าซื้อ (มาตรา ๕๗๒ - ๕๗๔)": (572, 574),
                    "ลักษณะ ๖ จ้างแรงงาน (มาตรา ๕๗๕ - ๕๘๖)": (575, 586),
                    "ลักษณะ ๗ จ้างทำของ (มาตรา ๕๘๗ - ๖๐๗)": (587, 607),
                    "ลักษณะ ๘ รับขน (มาตรา ๖๐๘ - ๖๓๙)": (608, 639),
                    "ลักษณะ ๙ ยืม (มาตรา ๖๔๐ - ๖๕๖)": {
                        "หมวด ๑ ยืมใช้คงรูป (มาตรา ๖๔๐ - ๖๔๙)": (640, 649),
                        "หมวด ๒ ยืมใช้สิ้นเปลือง (มาตรา ๖๕๐ - ๖๕๖)": (650, 656)
                    },
                    "ลักษณะ ๑๐ ฝากทรัพย์ (มาตรา ๖๕๗ - ๖๗๙)": {
                        "หมวด ๑ บทเบ็ดเสร็จทั่วไป (มาตรา ๖๕๗ - ๖๗๑)": (657, 671),
                        "หมวด ๒ วิธีเฉพาะการฝากเงิน (มาตรา ๖๗๒ - ๖๗๓)": (672, 673),
                        "หมวด ๓ วิธีเฉพาะสำหรับเจ้าสำนักโรงแรม (มาตรา ๖๗๔ - ๖๗๙)": (674, 679)
                    },
                    "ลักษณะ ๑๑ ค้ำประกัน (มาตรา ๖๘๐ - ๗๐๑)": {
                        "หมวด ๑ บทเบ็ดเสร็จทั่วไป (มาตรา ๖๘๐ - ๖๘๕/๑)": (680.0, 685.1),
                        "หมวด ๒ ผลก่อนชำระหนี้ (มาตรา ๖๘๖ - ๖๙๒)": (686, 692),
                        "หมวด ๓ ผลภายหลังชำระหนี้ (มาตรา ๖๙๓ - ๖๙๗)": (693, 697),
                        "หมวด ๔ ความระงับสิ้นไปแห่งการค้ำประกัน (มาตรา ๖๙๘ - ๗๐๑)": (698, 701)
                    },
                    "ลักษณะ ๑๒ จำนอง (มาตรา ๗๐๒ - ๗๔๖)": {
                        "หมวด ๑ บทเบ็ดเสร็จทั่วไป (มาตรา ๗๐๒ - ๗๑๔/๑)": (702.0, 714.1),
                        "หมวด ๒ สิทธิจำนองครอบเพียงใด (มาตรา ๗๑๕ - ๗๒๑)": (715, 721),
                        "หมวด ๓ สิทธิและหน้าที่ของผู้รับจำนองและผู้จำนอง (มาตรา ๗๒๒ - ๗๒๗/๑)": (722.0, 727.1),
                        "หมวด ๔ การบังคับจำนอง (มาตรา ๗๒๘ - ๗๓๕)": (728, 735),
                        "หมวด ๕ สิทธิและหน้าที่ของผู้รับโอนทรัพย์ซึ่งจำนอง (มาตรา ๗๓๖ - ๗๔๓)": (736, 743),
                        "หมวด ๖ ความระงับสิ้นไปแห่งสัญญาจำนอง (มาตรา ๗๔๔ - ๗๔๖)": (744, 746)
                    },
                    "ลักษณะ ๑๓ จำนำ (มาตรา ๗๔๗ - ๗๖๙)": {
                        "หมวด ๑ บทเบ็ดเสร็จทั่วไป (มาตรา ๗๔๗ - ๗๕๗)": (747, 757),
                        "หมวด ๒ สิทธิและหน้าที่ของผู้จำนำและผู้รับจำนำ (มาตรา ๗๕๘ - ๗๖๓)": (758, 763),
                        "หมวด ๓ การบังคับจำนำ (มาตรา ๗๖๔ - ๗๖๘)": (764, 768),
                        "หมวด ๔ ความระงับสิ้นไปแห่งการจำนำ (มาตรา ๗๖๙)": (769, 769)
                    },
                    "ลักษณะ ๑๔ เก็บของในคลังสินค้า (มาตรา ๗๗๐ - ๗๙๖)": {
                        "หมวด ๑ บทเบ็ดเสร็จทั่วไป (มาตรา ๗๗๐ - ๗๗๔)": (770, 774),
                        "หมวด ๒ ใบรับของคลังสินค้าและประทวนสินค้า (มาตรา ๗๗๕ - ๗๙๖)": (775, 796)
                    },
                    "ลักษณะ ๑๕ ตัวแทน (มาตรา ๗๙๗ - ๘๔๔)": {
                        "หมวด ๑ บทเบ็ดเสร็จทั่วไป (มาตรา ๗๙๗ - ๘๐๖)": (797, 806),
                        "หมวด ๒ หน้าที่และความรับผิดของตัวแทนต่อตัวการ (มาตรา ๘๐๗ - ๘๑๔)": (807, 814),
                        "หมวด ๓ หน้าที่และความรับผิดของตัวการต่อตัวแทน (มาตรา ๘๑๕ - ๘๑๙)": (815, 819),
                        "หมวด ๔ ความรับผิดของตัวการและตัวแทนต่อบุคคลภายนอก (มาตรา ๘๒๐ - ๘๒๕)": (820, 825),
                        "หมวด ๕ ความระงับสิ้นไปแห่งสัญญาตัวแทน (มาตรา ๘๒๖ - ๘๓๒)": (826, 832),
                        "หมวด ๖ ตัวแทนค้าต่าง (มาตรา ๘๓๓ - ๘๔๔)": (833, 844)
                    },
                    "ลักษณะ ๑๖ นายหน้า (มาตรา ๘๔๕ - ๘๔๙)": (845, 849),
                    "ลักษณะ ๑๗ ประนีประนอมยอมความ (มาตรา ๘๕๐ - ๘๕๒)": (850, 852),
                    "ลักษณะ ๑๘ การพนัน และขันต่อ (มาตรา ๘๕๓ - ๘๕๕)": (853, 855),
                    "ลักษณะ ๑๙ บัญชีเดินสะพัด (มาตรา ๘๕๖ - ๘๖๐)": (856, 860),
                    "ลักษณะ ๒๐ ประกันภัย (มาตรา ๘๖๑ - ๘๙๗)": {
                        "หมวด ๑ บทเบ็ดเสร็จทั่วไป (มาตรา ๘๖๑ - ๘๖๘)": (861, 868),
                        "หมวด ๒ ประกันวินาศภัย (มาตรา ๘๖๙ - ๘๘๘)": (869, 888),
                        "หมวด ๓ ประกันชีวิต (มาตรา ๘๘๙ - ๘๙๗)": (889, 897)
                    },
                    "ลักษณะ ๒๑ ตั๋วเงิน (มาตรา ๘๙๘ - ๑๐๑๑)": {
                        "หมวด ๑ บทเบ็ดเสร็จทั่วไป (มาตรา ๘๙๘ - ๙๐๗)": (898, 907),
                        "หมวด ๒ ตั๋วแลกเงิน (มาตรา ๙๐๘ - ๙๘๑)": (908, 981),
                        "หมวด ๓ ตั๋วสัญญาใช้เงิน (มาตรา ๙๘๒ - ๙๘๖)": (982, 986),
                        "หมวด ๔ เช็ค (มาตรา ๙๘๗ - ๑๐๐๐)": (987, 1000),
                        "หมวด ๕ อายุความ (มาตรา ๑๐๐๑ - ๑๐๐๕)": (1001, 1005),
                        "หมวด ๖ ตั๋วเงินปลอม ตั๋วเงินถูกลัก และตั๋วเงินหาย (มาตรา ๑๐๐๖ - ๑๐๑๑)": (1006, 1011)
                    },
                    "ลักษณะ ๒๒ หุ้นส่วนและบริษัท (มาตรา ๑๐๐๒ - ๑๒๗๓/๔)": {
                        "หมวด ๑ บทเบ็ดเสร็จทั่วไป (มาตรา ๑๐๑๒ - ๑๐๒๔)": (1012, 1024),
                        "หมวด ๒ ห้างหุ้นส่วนสามัญ (มาตรา ๑๐๒๕ - ๑๐๗๖)": (1025, 1076),
                        "หมวด ๓ ห้างหุ้นส่วนจำกัด (มาตรา ๑๐๗๗ - ๑๐๙๕)": (1077, 1095),
                        "หมวด ๔ บริษัทจำกัด (มาตรา ๑๐๙๖ - ๑๒๔๖/๗)": (1096.0, 1246.7),
                        "หมวด ๕ การชำระบัญชีห้างหุ้นส่วนจดทะเบียน ห้างหุ้นส่วนจำกัด และบริษัทจำกัด (มาตรา ๑๒๔๗ - ๑๒๗๓)": (
                            1247, 1273),
                        "หมวด ๖ การถอนทะเบียนห้างหุ้นส่วนจดทะเบียน ห้างหุ้นส่วนจำกัด และบริษัทจำกัดร้าง (มาตรา ๑๒๗๓/๑ - ๑๒๗๓/๔)": (
                            1273.1, 1273.4)
                    },
                    "ลักษณะ ๒๓ สมาคม (มาตรา ๑๒๗๔ - ๑๒๙๗)": (1274, 1297),
                },
                "บรรพ ๔ ทรัพย์สิน (มาตรา ๑๒๙๘ - ๑๔๓๔)": {
                    "ลักษณะ ๑ บทเบ็ดเสร็จทั่วไป (มาตรา ๑๒๙๘ - ๑๓๐๗)": (1298, 1307),
                    "ลักษณะ ๒ กรรมสิทธิ์ (มาตรา ๑๓๐๘ - ๑๓๖๖)": {
                        "หมวด ๑ การได้มาซึ่งกรรมสิทธิ์ (มาตรา ๑๓๐๘ - ๑๓๓๔)": (1308, 1334),
                        "หมวด ๒ แดนแห่งกรรมสิทธิ์ และการใช้กรรมสิทธิ์ (มาตรา ๑๓๓๕ - ๑๓๕๕)": (1335, 1355),
                        "หมวด ๓ กรรมสิทธิ์รวม (มาตรา ๑๓๕๖ - ๑๓๖๖)": (1356, 1366)
                    },
                    "ลักษณะ ๓ ครอบครอง (มาตรา ๑๓๖๗ - ๑๓๘๖)": (1367, 1386),
                    "ลักษณะ ๔ ภาระจำยอม (มาตรา ๑๓๘๗ - ๑๔๐๑)": (1387, 1401),
                    "ลักษณะ ๕ อาศัย (มาตรา ๑๔๐๒ - ๑๔๐๙)": (1402, 1409),
                    "ลักษณะ ๖ สิทธิเหนือพื้นดิน (มาตรา ๑๔๑๐ - ๑๔๑๖)": (1410, 1416),
                    "ลักษณะ ๗ สิทธิเก็บกิน (มาตรา ๑๔๑๗ - ๑๔๒๘)": (1417, 1428),
                    "ลักษณะ ๘ ภาระติดพันในอสังหาริมทรัพย์ (มาตรา ๑๔๒๙ - ๑๔๓๔)": (1429, 1434),
                },
                "บรรพ ๕ ครอบครัว (มาตรา ๑๔๓๕ - ๑๕๙๘/๔๑)": {
                    "ลักษณะ ๑ การสมรส (มาตรา ๑๔๓๕ - ๑๕๓๕)": {
                        "หมวด ๑ การหมั้น (มาตรา ๑๔๓๕ - ๑๔๔๗/๒)": (1435.0, 1447.2),
                        "หมวด ๒ เงื่อนไขแห่งการสมรส (มาตรา ๑๔๔๘ - ๑๔๖๐)": (1448, 1460),
                        "หมวด ๓ ความสัมพันธ์ระหว่างสามีภริยา (มาตรา ๑๔๖๑ - ๑๔๖๔/๑)": (1461.0, 1464.1),
                        "หมวด ๔ ทรัพย์สินระหว่างสามีภริยา (มาตรา ๑๔๖๕ - ๑๔๙๓)": (1465, 1493),
                        "หมวด ๕ ความเป็นโมฆะของการสมรส (มาตรา ๑๔๙๔ - ๑๕๐๐)": (1494, 1500),
                        "หมวด ๖ การสิ้นสุดแห่งการสมรส (มาตรา ๑๕๐๑ - ๑๕๓๕)": (1501, 1535),
                    },
                    "ลักษณะ ๒ บิดามารดากับบุตร (มาตรา ๑๕๓๖ - ๑๕๙๘/๓๗)": {
                        "หมวด ๑ บิดามารดา (มาตรา ๑๕๓๖ - ๑๕๖๐)": (1536, 1560),
                        "หมวด ๒ สิทธิและหน้าที่ของบิดามารดาและบุตร (มาตรา ๑๕๖๑ - ๑๕๘๔/๑)": (1561.0, 1584.1),
                        "หมวด ๓ ความปกครอง (มาตรา ๑๕๘๕ - ๑๕๙๘/๑๘)": (1585.00, 1598.9),
                        "หมวด ๔ บุตรบุญธรรม (มาตรา ๑๕๙๘/๑๙ - ๑๕๙๘/๓๗)": (1598.19, 1598.37)
                    },
                    "ลักษณะ ๓ ค่าอุปการะเลี้ยงดู (มาตรา ๑๕๙๘/๓๘ - ๑๕๙๘/๔๑)": (1598.38, 1598.41),
                },
                "บรรพ ๖ มรดก (มาตรา ๑๕๙๙ - ๑๗๕๕)": {
                    "ลักษณะ ๑ บทเบ็ดเสร็จทั่วไป (มาตรา ๑๕๙๙ - ๑๖๑๙)": {
                        "หมวด ๑ การตกทอดแห่งทรัพย์มรดก (มาตรา ๑๕๙๙ - ๑๖๐๓)": (1599, 1603),
                        "หมวด ๒ การเป็นทายาท (มาตรา ๑๖๐๔ - ๑๖๐๗)": (1604, 1607),
                        "หมวด ๓ การตัดมิให้รับมรดก (มาตรา ๑๖๐๘ - ๑๖๐๙)": (1608, 1609),
                        "หมวด ๔ การสละมรดกและอื่น ๆ (มาตรา ๑๖๑๐ - ๑๖๑๙)": (1610, 1619)
                    },
                    "ลักษณะ ๒ สิทธิโดยธรรมในการรับมรดก (มาตรา ๑๖๒๐ - ๑๖๔๕)": {
                        "หมวด ๑ บทเบ็ดเสร็จทั่วไป (มาตรา ๑๖๒๐ - ๑๖๒๘)": (1620, 1628),
                        "หมวด ๒ การแบ่งทรัพย์มรดกระหว่างทายาทโดยธรรมในลำดับและชั้นต่าง ๆ (มาตรา ๑๖๒๙ - ๑๖๓๑)": (
                            1629, 1631),
                        "หมวด ๓ การแบ่งส่วนมรดกของทายาทโดยธรรมในลำดับและชั้นต่าง ๆ (มาตรา ๑๖๓๒ - ๑๖๓๘)": (1632, 1638),
                        "หมวด ๔ การรับมรดกแทนที่กัน (มาตรา ๑๖๓๙ - ๑๖๔๕)": (1639, 1645)
                    },
                    "ลักษณะ ๓ พินัยกรรม (มาตรา ๑๖๔๖ - ๑๗๑๐)": {
                        "หมวด ๑ บทเบ็ดเสร็จทั่วไป (มาตรา ๑๖๔๖ - ๑๖๕๔)": (1646, 1654),
                        "หมวด ๒ แบบพินัยกรรม (มาตรา ๑๖๕๕ - ๑๖๗๒)": (1655, 1672),
                        "หมวด ๓ ผลและการตีความแห่งพินัยกรรม (มาตรา ๑๖๗๓ - ๑๖๘๕)": (1673, 1685),
                        "หมวด ๔ พินัยกรรมที่ตั้งผู้ปกครองทรัพย์ (มาตรา ๑๖๘๖ - ๑๖๙๒)": (1686, 1692),
                        "หมวด ๕ การเพิกถอนและการตกไปแห่งพินัยกรรมหรือข้อกำหนดพินัยกรรม (มาตรา ๑๖๙๓ - ๑๖๙๙)": (
                            1693, 1699),
                        "หมวด ๖ ความเสียเปล่าแห่งพินัยกรรมหรือข้อกำหนดพินัยกรรม (๑๗๐๐ - ๑๗๑๐)": (1700, 1710)
                    },
                    "ลักษณะ ๔ วิธีจัดการและปันทรัพย์มรดก (มาตรา ๑๗๑๑ - ๑๗๕๒)": {
                        "หมวด ๑ ผู้จัดการมรดก (มาตรา ๑๗๑๑ - ๑๗๓๓)": (1711, 1733),
                        "หมวด ๒ การรวบรวมจำหน่ายทรัพย์มรดกเป็นตัวเงินและการชำระหนี้กับแบ่งปันทรัพย์มรดก (มาตรา ๑๗๓๔ - ๑๗๔๔)": (
                            1734, 1744),
                        "หมวด ๓ การแบ่งมรดก (มาตรา ๑๗๔๕ - ๑๗๕๒)": (1745, 1752)
                    },
                    "ลักษณะ ๕ มรดกที่ไม่มีผู้รับ (มาตรา ๑๗๕๓)": (1753),
                    "ลักษณะ ๖ อายุความ (มาตรา ๑๗๕๔ - ๑๗๕๕)": (1754, 1755),
                },
                "เหตุผลในการประกาศใช้": None,
                "อื่นๆ": None
            }

            selected_section = st.selectbox("เลือกบรรพ", list(sections.keys()), key="section_unique")

            # กรณีเลือก พระราชบัญญัติ หรือ พระราชกฤษฎีกา หรือ เหตุผลในการประกาศใช้
            if selected_section in ["พระราชบัญญัติให้ใช้ประมวลกฎหมายแพ่งและพาณิชย์", "เหตุผลในการประกาศใช้", "อื่นๆ"]:
                selected_sub_section = None
                selected_category = None
                section_number = None
                if selected_section == "พระราชบัญญัติให้ใช้ประมวลกฎหมายแพ่งและพาณิชย์":
                    selected_sub_section = st.selectbox("เลือกรายการ", list(sections[selected_section].keys()),
                                                        key="sub_section_unique")
            else:
                # ตรวจสอบกรณี "ข้อความเบื้องต้น (มาตรา ๑ - ๓)"
                if selected_section == "ข้อความเบื้องต้น (มาตรา ๑ - ๓)":
                    min_section, max_section = sections[selected_section]
                    section_number = st.number_input(
                        f"มาตรา ({min_section} - {max_section})",
                        min_value=int(min_section),
                        max_value=int(max_section),
                        format="%d",
                        key="section_number_unique"
                    )
                    selected_sub_section = None
                    selected_category = None
                else:
                    selected_sub_section = st.selectbox("เลือกลักษณะ", list(sections[selected_section]),
                                                        key="sub_section_unique")
                    if isinstance(sections[selected_section][selected_sub_section], dict):
                        selected_category = st.selectbox(
                            "เลือกหมวด",
                            list(sections[selected_section][selected_sub_section].keys()),
                            key="category_unique"
                        )
                        min_section, max_section = sections[selected_section][selected_sub_section][selected_category]
                    else:
                        selected_category = None
                        value = sections[selected_section][selected_sub_section]
                        min_section, max_section = value if isinstance(value, tuple) else (value, value)

                    section_number = st.number_input(
                        f"มาตรา ({min_section} - {max_section})",
                        min_value=float(min_section) if isinstance(min_section, float) else int(min_section),
                        max_value=float(max_section) if isinstance(max_section, float) else int(max_section),
                        step=0.01 if isinstance(min_section, float) else 1,
                        format="%.2f" if isinstance(min_section, float) else "%d",
                        key="section_number_unique"
                    )

            law_content = st.text_area("เนื้อหากฎหมาย", height=400, key="law_content_unique")

            if st.button("Submit", key="submit_law_btn_unique"):
                if not section_number and selected_section not in ["พระราชบัญญัติให้ใช้ประมวลกฎหมายแพ่งและพาณิชย์",
                                                                   "เหตุผลในการประกาศใช้"]:
                    st.error("กรุณากรอกข้อมูลในช่อง มาตรา")
                elif not law_content:
                    st.error("กรุณากรอกเนื้อหากฎหมาย")
                else:
                    st.success(
                        f"กฎหมายที่บันทึก:\nประเภท: {selected_law_type}\nบรรพ: {selected_section}\nมาตรา: {section_number or 'N/A'}\nเนื้อหากฎหมาย: {law_content}")
                    with st.spinner('กำลังฝังข้อมูล...'):
                        try:
                            store_law_data_in_mongo(
                                selected_law_type,
                                draft_date,
                                section_number,
                                law_content,
                                selected_section,
                                selected_sub_section,
                                selected_category
                            )
                            st.success("ฝังข้อมูลสำเร็จและบันทึกลง MongoDB แล้ว!")
                        except Exception as e:
                            st.error(f"เกิดข้อผิดพลาด: {str(e)}")

        # กรณีเลือก "พรบ คอมพิวเตอร์"
        if selected_law_type == "พรบ คอมพิวเตอร์":
            # ให้กรอกวันที่ร่าง
            draft_date = st.date_input("วันที่ร่าง", datetime.now(), key="draft_date")

            # ดรอปดาวน์หัวข้อที่มีตัวเลือก "อื่นๆ"
            additional_option = st.selectbox("เลือกหัวข้อ", ["มาตรามาตรฐาน", "อื่นๆ"], key="additional_option")

            # ถ้าเลือก "อื่นๆ" จะไม่แสดงฟิลด์สำหรับกรอกมาตรา
            if additional_option != "อื่นๆ":
                law_section = st.text_input("มาตรา", key="law_section")  # แสดงฟิลด์มาตราเฉพาะเมื่อไม่ใช่ "อื่นๆ"
                selected_section = additional_option  # ใช้หัวข้อที่เลือกจากดรอปดาวน์
            else:
                selected_section = "อื่นๆ"  # กำหนดเป็น "อื่นๆ" โดยไม่แสดงฟิลด์ "มาตรา"

            # กรอกเนื้อหากฎหมาย
            law_content = st.text_area("เนื้อหากฎหมาย", height=400, key="law_content_unique")

            if st.button("Submit", key="submit_law_btn_unique"):
                if not law_content:
                    st.error("กรุณากรอกเนื้อหากฎหมาย")
                else:
                    # แสดงข้อมูลที่บันทึก
                    st.success(
                        f"กฎหมายที่บันทึก:\nประเภท: {selected_law_type}\nหัวข้อ: {selected_section}\n"
                        f"มาตรา: {law_section if additional_option != 'อื่นๆ' and law_section else 'N/A'}\nเนื้อหากฎหมาย: {law_content}"
                    )
                    with st.spinner('กำลังฝังข้อมูล...'):
                        try:
                            # ส่งข้อมูลทั้งหมดไปยัง MongoDB
                            store_law_data_in_mongo(selected_law_type, draft_date,
                                                    law_section if additional_option != "อื่นๆ" else None, law_content,
                                                    selected_section)
                            st.success("ฝังข้อมูลสำเร็จและบันทึกลง MongoDB แล้ว!")
                        except Exception as e:
                            st.error(f"เกิดข้อผิดพลาด: {str(e)}")

        elif selected_law_type == "PDPA":
            # เงื่อนไขสำหรับ "PDPA" ทำงานเหมือน "พรบ คอมพิวเตอร์"
            draft_date = st.date_input("วันที่ร่าง", datetime.now(), key="draft_date_pdpa")
            additional_option = st.selectbox("เลือกหัวข้อ", ["มาตรามาตรฐาน", "อื่นๆ"], key="additional_option_pdpa")

            if additional_option != "อื่นๆ":
                law_section = st.text_input("มาตรา", key="law_section_pdpa")
                selected_section = additional_option
            else:
                selected_section = "อื่นๆ"
                law_section = None

            law_content = st.text_area("เนื้อหากฎหมาย", height=400, key="law_content_pdpa")

            if st.button("Submit", key="submit_law_btn_pdpa"):
                if not law_content:
                    st.error("กรุณากรอกเนื้อหากฎหมาย")
                else:
                    st.success(
                        f"กฎหมายที่บันทึก:\nประเภท: {selected_law_type}\nหัวข้อ: {selected_section}\n"
                        f"มาตรา: {law_section if law_section else 'N/A'}\nเนื้อหากฎหมาย: {law_content}"
                    )
                    with st.spinner('กำลังฝังข้อมูล...'):
                        try:
                            store_law_data_in_mongo(selected_law_type, draft_date, law_section, law_content,
                                                    selected_section)
                            st.success("ฝังข้อมูลสำเร็จและบันทึกลง MongoDB แล้ว!")
                        except Exception as e:
                            st.error(f"เกิดข้อผิดพลาด: {str(e)}")

        # ตรวจสอบว่าถ้าเลือก "เคสตัวอย่าง" ให้แสดงเฉพาะช่องกรอกเนื้อหากฎหมาย

        if selected_law_type in ["เคสตัวอย่าง", "พรบ การจัดซื้อจัดจ้าง", "พรบ ส่งเสริมและรักษาสิ่งแวดล้อม"]:
            # ตรวจสอบว่าผู้ใช้ได้ใส่วันที่ร่างหรือยัง
            draft_date = st.date_input("วันที่ร่าง", datetime.now(), key=f"draft_date_{selected_law_type}")
            # ช่องกรอกเนื้อหากฎหมาย
            law_content = st.text_area("เนื้อหากฎหมาย", height=400, key=f"law_content_{selected_law_type}")

            # ปุ่มสำหรับบันทึก
            if st.button("Submit", key=f"submit_{selected_law_type}"):
                if not law_content:
                    st.error("กรุณากรอกเนื้อหากฎหมาย")
                else:
                    st.success(f"{selected_law_type} ที่บันทึก:\nเนื้อหากฎหมาย: {law_content}")
                    with st.spinner('กำลังฝังข้อมูล...'):
                        try:
                            # เรียกใช้ฟังก์ชัน store_law_data_in_mongo เพื่อบันทึกข้อมูลและสร้าง embedding
                            store_law_data_in_mongo(selected_law_type, draft_date, None, law_content, selected_law_type)
                            st.success("ฝังข้อมูลสำเร็จและบันทึกลง MongoDB แล้ว!")
                        except Exception as e:
                            st.error(f"เกิดข้อผิดพลาด: {str(e)}")

    with tab3:
        st.subheader("Change Status")

        # ดึงรายการเอกสารจาก MongoDB มาแสดงใน dropdown
        document_list = get_document_list()

        if document_list:
            # เปลี่ยนช่องใส่ Document ID เป็น dropdown ให้ผู้ใช้เลือก
            selected_document = st.selectbox("เลือกเอกสารที่ต้องการเปลี่ยนสถานะ", document_list,
                                             format_func=lambda x: x[1])

            # เมื่อเลือกเอกสาร ให้ดึงข้อมูลเอกสารมาแสดง
            document_data = get_document_by_id(selected_document[0])

            if document_data:
                # แสดงข้อมูลเอกสาร
                st.write(f"ประเภท: {document_data.get('ประเภทกฎหมาย', 'N/A')}")
                st.write(f"ภาค: {document_data.get('ภาค', 'N/A')}")
                st.write(f"มาตรา: {document_data.get('มาตรา', 'N/A')}")
                st.write(f"เนื้อหา: {document_data.get('chunk_content', 'N/A')}")
                st.write(f"สถานะปัจจุบัน: {document_data.get('status', 'N/A')}")

                # ปุ่มเปลี่ยนสถานะ พร้อม key ที่ไม่ซ้ำกัน
                if st.button("Toggle Status", key="toggle_status_button"):
                    toggle_document_status(selected_document[0])  # ส่งเฉพาะ Document ID

                    # ดึงข้อมูลใหม่หลังจากเปลี่ยนสถานะแล้ว
                    updated_document_data = get_document_by_id(selected_document[0])

                    if updated_document_data:
                        st.success("เปลี่ยนสถานะเรียบร้อย!")
                        # แสดงข้อมูลใหม่หลังเปลี่ยนสถานะ
                        st.write(f"ประเภท: {updated_document_data.get('ประเภทกฎหมาย', 'N/A')}")
                        st.write(f"ภาค: {updated_document_data.get('ภาค', 'N/A')}")
                        st.write(f"มาตรา: {updated_document_data.get('มาตรา', 'N/A')}")
                        st.write(f"เนื้อหา: {updated_document_data.get('chunk_content', 'N/A')}")
                        st.write(f"สถานะปัจจุบัน: {updated_document_data.get('status', 'N/A')}")
            else:
                st.write("ไม่พบข้อมูลเอกสาร")
        else:
            st.write("ไม่พบเอกสาร")


if __name__ == "__main__":
    main()
