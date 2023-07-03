# Classificação de imagem utilizando algoritmo de DeepLearning com Teachable Machine
<h2>Esse modelo deep learnign visa classificar a qualidade da imagem como Boa ou Ruim. Na base de dados, contém somente 4 imagens, mas de valia para realizar a classificação.</h2>
<h3>Para a configuração e instalação do projeto, é necessário: </h3>
<ol>
  <li>
    Ter instalado a linguagem PYTHON no seu computador
  </li>
  <li>
    Instalado as seguintes bibliotecas:
  <ol>
    <li>
    tensorflow -> pip install tensorflow
  </li>
  <li>
    keras -> pip install keras
  </li>
  <li>
    pillow -> pip install pillow
  </li> </ol>
  </li>
</ol>

O modelo foi criado utilizando Teachable machine e a base de dados contém somente apenas 4 imagens, visto que é para um exemplo básico.
Para testar com imagens contidas na BD, é necessário alterar o caminho mencionado abaixo. 
```
image = Image.open("images/bad_image.jpg").convert("RGB")
```


Após configurar todo o projeto e rodar, será exibido a seguinte mensagem abaixo com o resultado e acurácia. 

```
1/1 [==============================] - 1s 1s/step
Class: Bad
Confidence Score: 0.9680495
```
