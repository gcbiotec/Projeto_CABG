# tensorflow - cpu == 2.5.2
# keras == 2.4.0

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

#from keras.models import model_from_json

st.header('Calule o risco do paciente!')

st.markdown('---')
st.markdown('### Entrada das comorbidades seu paciente para o modelo:')
st.markdown(' ')

col1, col2 = st.columns(2)

x1 = col1.radio('Sexo do paciente: 1 = Masc / 0 = Fem', ['1', '0'])
x2 = col1.slider('Idade', 20, 95, 40, 1, help = 'Qual a sua idade?')
x3 = col1.radio('Tem Diabetes Melittus? 1 = Sim / 0 = Não', ['1','0'])
x4 = col1.radio('IAM prévio? 1 = Sim / 0 = Nao', ['1','0'])
x5 = col1.radio('Angina? 1 = Sim / 0 = Nao', ['1','0'])
x6 = col1.slider('Valor de Clearence', 50, 150, 50, 1, help = 'Clearence de Creatinina')

x7 = col2.radio('Doença Cerebro-Vascular? 1 = Sim / 0 = Não', ['1','0'])
x8 = col2.radio('FA prévia: 1 = Sim / 0 = Não', ['1','0'])
x9 = col2.radio('Tem Anemia? 1 = Sim / 0 = Não', ['1','0'])
x10 = col2.radio('Tem cirurgia prévia? 1 = Sim / 0 = Não', ['1','0'])
x11 = col2.radio('Tem cirurgia de urgencia? 1 = Sim / 0 = Não', ['1','0'])
x12 = col2.radio('Teve uso de CEC na cirurgia? 1 = Sim / 0 = Não', ['1','0'])

st.markdown('---')

dicionario = {'sexo': [x1],
              'idade': [x2],
              'dm': [x3],
              'iam_rec': [x4],
              'angina_inst': [x5],
              'clearence': [x6],
              'dcv_pre': [x7],
              'fa_pre': [x8],
              'anemia': [x9],
              'cirCVpre': [x10],
              'urgencia': [x11],
              'cec': [x12],
              }

dados = pd.DataFrame(dicionario)
dados = np.asarray(dados).astype('int64')

# st.write(dados)

st.markdown('---')

json_file = open('modelo_noReLu_18.01.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

#loaded_model = model_from_json(loaded_model_json)
loaded_model = tf.keras.models.model_from_json(loaded_model_json)

loaded_model.load_weights('modelo_noReLu_18.01.h5')

loaded_model.compile(optimizer='Nadam', loss='categorical_crossentropy',
               metrics=['accuracy'])

# with open('modelo_RNN_tensorflow.pkl', 'rb') as f:
#     modelo = pickle.load(f)
#     saida = predict_model(modelo, dados)

saida_final = 'null'

if st.button('EXECUTAR O MODELO!'):
    saida = loaded_model.predict(dados)
#   classificacao = int(saida['TenYearCHD'])

    saida_final = [np.argmax(t) for t in saida]

# st.write(saida_final)

if saida_final == [0]:
        st.write('Paciente não tem previsão de óbito!')
elif saida_final == [1]:
        st.write('Paciente tem chance de óbito')
else:
        st.write('Preencha os dados e aperte no botão acima!')
