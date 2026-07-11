import streamlit as st
import tensorflow as tf
import json
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Configuração da página do Streamlit
st.set_page_config(page_title="Detector de Spam AI", page_icon="🚨", layout="centered")

# IMPORTANTE: Deve ser exatamente o mesmo valor usado no novo script de treino (Passo 1)
MAX_LEN = 80

# Cache para carregar o modelo apenas uma vez e economizar memória no Render
@st.cache_resource
def carregar_arquivos():
    # Carrega o modelo Keras treinado robusto
    model = tf.keras.models.load_model('modelo_spam.h5')
    
    # Carrega o tokenizador ajustado ao dataset real
    with open('tokenizer.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
        tokenizer = tokenizer_from_json(data)
        
    return model, tokenizer

# Tratamento de erro caso os arquivos não existam localmente ou no repositório
try:
    model, tokenizer = carregar_arquivos()
except Exception as e:
    st.error("Erro ao carregar o modelo. Certifique-se de ter rodado o novo treino e colocado os arquivos 'modelo_spam.h5' e 'tokenizer.json' na mesma pasta deste script.")
    st.stop()

# --- Interface Gráfica ---

st.title("🚨 Detector de Mensagens de Spam")
st.write("Insira o texto da mensagem abaixo para verificar se ela é segura ou um potencial Spam.")

# Área para o usuário digitar a mensagem
user_input = st.text_area("Mensagem:", placeholder="Digite ou cole a mensagem aqui...", height=150)

# Botão de gatilho para acionar a classificação
if st.button("Analisar Mensagem", type="primary"):
    if user_input.strip() == "":
        st.warning("Por favor, digite alguma mensagem antes de analisar.")
    else:
        # 1. Transforma o texto em sequência numérica com base no tokenizador treinado
        sequence = tokenizer.texts_to_sequences([user_input])
        # 2. Aplica o padding para que o vetor tenha o tamanho exato de 80 posições
        padded = pad_sequences(sequence, maxlen=MAX_LEN, padding='post', truncating='post')
        
        # 3. Executa a inferência com o TensorFlow
        predicao = model.predict(padded)[0][0]
        
        st.subheader("Resultado da Análise:")
        
        # 4. Define o veredito com base no limite clássico de 0.5 (50%)
        if predicao >= 0.5:
            st.error(f"⚠️ **Esta mensagem é SPAM!** (Confiança: {predicao*100:.2f}%)")
        else:
            st.success(f"✅ **Mensagem Segura.** (Confiança de ser Spam: {predicao*100:.2f}%)")

st.markdown("---")
st.caption("Sistema de Processamento de Linguagem Natural desenvolvido com TensorFlow e Streamlit.")