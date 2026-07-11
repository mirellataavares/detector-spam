import json
import urllib.request
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

print("Baixando dataset real de spam...")
# Baixa o dataset oficial de SMS Spam
url = "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/sms.tsv"
df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])

# Como o dataset é em inglês, vamos adicionar manualmente alguns exemplos em português 
# para o modelo entender o nosso idioma também!
exemplos_pt = [
    ('ham', 'Oi mãe, tudo bem? Me liga quando puder.'),
    ('ham', 'Fala cara, beleza? Me passa o contato do fornecedor da reunião.'),
    ('ham', 'Vamos jogar futebol amanhã à noite?'),
    ('spam', 'Ganhe dinheiro rápido trabalhando de casa! Clique aqui'),
    ('spam', 'PARABÉNS! Você ganhou um prêmio de 5000 reais. Resgate agora'),
    ('spam', 'Urgente: sua conta será bloqueada. Acesse o link para atualizar.')
]
df_pt = pd.DataFrame(exemplos_pt, columns=['label', 'message'])
df = pd.concat([df, df_pt], ignore_index=True)

# Converte labels para números: spam = 1, ham = 0
df['label'] = df['label'].map({'spam': 1, 'ham': 0})

mensagens = df['message'].astype(str).values
labels = df['label'].values

# Configurações do Tokenizador
max_words = 5000  # Aumentamos o vocabulário
max_len = 80     # Aumentamos o tamanho máximo da frase

tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(mensagens)
sequences = tokenizer.texts_to_sequences(mensagens)
padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

# Arquitetura ligeiramente melhorada para lidar com mais dados
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(max_words, 32, input_length=max_len),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print("Treinando o novo modelo (isso pode levar alguns segundos)...")
model.fit(padded_sequences, labels, epochs=15, batch_size=64, verbose=1)

# Salvar os novos arquivos
model.save('modelo_spam.h5')
tokenizer_json = tokenizer.to_json()
with open('tokenizer.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(tokenizer_json, ensure_ascii=False))

print("\n✨ Novo modelo robusto e Tokenizador salvos com sucesso!")