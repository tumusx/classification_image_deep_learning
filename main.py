from keras.models import load_model  # Importa a função load_model do Keras para carregar o modelo treinado
from PIL import Image, ImageOps  # Importa as classes Image e ImageOps da biblioteca PIL (Pillow)
import numpy as np  # Importa a biblioteca numpy para trabalhar com arrays numéricos
import matplotlib.pyplot as plt  # Importa a biblioteca matplotlib para criar gráficos

# Define as opções de impressão para desativar a notação científica
np.set_printoptions(suppress=True)

# Carrega o modelo treinado a partir do arquivo "model/keras_model.h5"
model = load_model("model/keras_model.h5", compile=False)

# Carrega os rótulos das classes a partir do arquivo "model/labels.txt"
class_names = open("model/labels.txt", "r").readlines()

# Cria o array de formato apropriado para alimentar o modelo do Keras
# O formato é (1, 224, 224, 3), onde 1 é o número de imagens, 224 é a largura e altura da imagem, e 3 é o número de canais de cor (RGB)
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Abre a imagem a ser classificada a partir do arquivo "images/bad_image.jpg" e converte para o modo RGB
image = Image.open("images/bad_image.jpg").convert("RGB")

# Redimensiona a imagem para ter no mínimo 224x224 pixels e faz um recorte a partir do centro
size = (224, 224)
image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

# Converte a imagem em um array do NumPy
image_array = np.asarray(image)

# Normaliza o array da imagem, convertendo-o para float32, dividindo por 127.5 e subtraindo 1
normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

# Carrega o array da imagem normalizada para o array 'data'
data[0] = normalized_image_array

# Faz a previsão usando o modelo
prediction = model.predict(data)

# Obtém o índice da classe com a maior probabilidade de previsão
index = np.argmax(prediction)

# Obtém o nome da classe correspondente ao índice encontrado
class_name = class_names[index]

# Obtém a pontuação de confiança da previsão
confidence_score = prediction[0][index]

# Cria um gráfico de linha para visualizar a acurácia
plt.plot([0, 1], [confidence_score, confidence_score], color='blue')

# Configurações do gráfico
plt.title('Acurácia')
plt.xlabel('Índice')
plt.ylabel('Valor')

# Definir limites dos eixos
plt.xlim(0, 1)
plt.ylim(min(confidence_score, 0), max(confidence_score, 1))

# Salva o gráfico em um arquivo
plt.savefig('grafico.png')

# Imprime a classe prevista e a pontuação de confiança
print("Class:", class_name[2:], end="")
print("Confidence Score:", confidence_score)
