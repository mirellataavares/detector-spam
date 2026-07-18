import streamlit as st
import tensorflow as tf
import json
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences


st.set_page_config(page_title="Detector de Spam AI", page_icon="🚨", layout="centered")


MAX_LEN = 80


@st.cache_resource
def carregar_arquivos():
   
    model = tf.keras.models.load_model('modelo_spam.h5')
    
    with open('tokenizer.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
        tokenizer = tokenizer_from_json(data)
        
    return model, tokenizer


try:
    model, tokenizer = carregar_arquivos()
except Exception as e:
    st.error("Erro ao carregar o modelo. Certifique-se de ter rodado o novo treino e colocado os arquivos 'modelo_spam.h5' e 'tokenizer.json' na mesma pasta deste script.")
    st.stop()


st.title("🚨 Detector de Mensagens de Spam")
st.write("Insira o texto da mensagem abaixo para verificar se ela é segura ou um potencial Spam.")


user_input = st.text_area("Mensagem:", placeholder="Digite ou cole a mensagem aqui...", height=150)


if st.button("Analisar Mensagem", type="primary"):
    if user_input.strip() == "":
        st.warning("Por favor, digite alguma mensagem antes de analisar.")
    else:
      
        sequence = tokenizer.texts_to_sequences([user_input])
       
        padded = pad_sequences(sequence, maxlen=MAX_LEN, padding='post', truncating='post')
        
       
        predicao = model.predict(padded)[0][0]
        
        st.subheader("Resultado da Análise:")
        
        
        if predicao >= 0.5:
            st.error(f"⚠️ **Esta mensagem é SPAM!** (Confiança: {predicao*100:.2f}%)")
        else:
            st.success(f"✅ **Mensagem Segura.** (Confiança de ser Spam: {predicao*100:.2f}%)")

st.markdown("---")
st.caption("Sistema de Processamento de Linguagem Natural desenvolvido com TensorFlow e Streamlit.")
