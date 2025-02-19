# Classificação de Escoliose utilizando TensorFlow

*To access this README in english, [click here](README.eng.md)*

Este projeto tem como objetivo classificar a gravidade da escoliose com base nos dados dos pacientes. Utilizando o TensorFlow, foi construído um modelo de Perceptron Multicamadas (MLP) para prever se um paciente possui escoliose (grau de escoliose > 10) ou não (grau de escoliose ≤ 10). O projeto utiliza o Keras Tuner para a otimização automatizada dos hiperparâmetros, embora nossos experimentos tenham revelado que, mesmo com um ajuste extenso, os modelos não conseguiram aprender padrões significativos a partir do conjunto de dados.

> **Nota:** Uma dissertação de mestrado aplicou aprendizado de máquina a este mesmo conjunto de dados e relatou uma acurácia de 75%. No entanto, essa dissertação não considerou que múltiplas amostras foram coletadas do mesmo indivíduo. Ao levar esse aspecto em consideração, nossos experimentos não produziram resultados consistentes ou significativos. Para mais detalhes, consulte o [documento da dissertação](https://repositorio.unesp.br/entities/publication/153798ea-a515-4313-8302-a8f59bf1e127).

## Sumário

1. [Contexto](#contexto)
2. [O que é um MLP?](#o-que-é-um-mlp)
3. [Informações sobre o Conjunto de Dados](#informações-sobre-o-conjunto-de-dados)
4. [Implementação com TensorFlow](#implementação-com-tensorflow)
5. [Otimização de Hiperparâmetros com Keras Tuner](#otimização-de-hiperparâmetros-com-keras-tuner)
6. [O Processo](#o-processo)
7. [Análise e Resultados](#análise-e-resultados)
8. [Conclusão](#conclusão)
9. [Contribuições](#contribuições)
10. [Contato](#contato)
11. [Estrutura do Repositório](#estrutura-do-repositório)

## Contexto

A escoliose é caracterizada por uma curvatura anormal da coluna vertebral, com o diagnóstico geralmente baseado no grau medido de escoliose. Neste projeto, utilizamos dados demográficos, físicos e biomecânicos dos pacientes — incluindo leituras dos sensores do baropodômetro — para classificar a severidade da escoliose. Embora trabalhos anteriores (como a dissertação de mestrado vinculada) tenham apresentado acurácias promissoras, esses estudos não consideraram que várias amostras podem ser coletadas do mesmo indivíduo. Ao levar esse fator em conta, nossos modelos consistentemente apresentaram dificuldades em aprender com os dados.

## O que é um MLP?

Um **Perceptron Multicamadas (MLP)** é um tipo de rede neural feedforward composto por:
- **Camada de Entrada:** Responsável por receber as características brutas dos dados.
- **Camadas Ocultas:** Conjunto de neurônios que aplicam somas ponderadas seguidas de funções de ativação (geralmente ReLU) para extrair padrões complexos.
- **Camada de Saída:** Responsável por gerar a previsão final; para tarefas de classificação binária, utiliza-se uma função de ativação sigmoide para fornecer probabilidades.

Os MLPs são treinados através da retropropagação, com o objetivo de minimizar uma função de perda (como a entropia cruzada binária), sendo especialmente indicados para o processamento de dados tabulares.

## Informações sobre o Conjunto de Dados

O conjunto de dados é composto por registros de pacientes contendo diversas características, tais como:

- **Id:** Identificador único para cada paciente.
- **Nome:** Identificador anônimo do paciente.
- **Idade:** Idade do paciente (em anos).
- **Massa:** Massa corporal do paciente (em kg).
- **Altura:** Altura do paciente (em metros).
- **Gênero:** Colunas binárias que indicam o gênero.
- **Lateralidade:** Indicadores para destros ou canhotos, embora existam colunas redundantes.
- **CoP_ML:** Medida do centro de pressão na direção medial-lateral.
- **Scolio:** Grau medido da escoliose.
- **Scolio_Class:** Rótulo binário derivado do grau de escoliose:
  - **0 (Sem Escoliose):** Grau de escoliose ≤ 10
  - **1 (Com Escoliose):** Grau de escoliose > 10

Além disso, o conjunto de dados inclui 120 características biomecânicas provenientes das leituras do baropodômetro, onde:
- As colunas de `s0` a `s59` representam as leituras do pé esquerdo.
- As colunas de `s60` a `s119` representam as leituras do pé direito.

Cada valor dos sensores representa a pressão média registrada durante um período de 30 a 60 segundos.

## Implementação com TensorFlow

O MLP foi implementado utilizando a API Keras do TensorFlow. Os passos principais da implementação incluíram:

1. **Preparação dos Dados:**
   - **Limpeza e Seleção de Características:**
     - Foram importados os dados e analisadas todas as características não relacionadas aos sensores para identificar correlações elevadas (maiores que 0,8).
     - Remoção de 12 características altamente correlacionadas, mantendo apenas a mais representativa.
     - Eliminação de características redundantes, como colunas duplicadas de lateralidade e gênero, além do identificador único.
   - **Análise Exploratória:**
     - Realização de análises exploratórias básicas utilizando boxplots e histogramas.
     - Observou-se que a distribuição etária é ampla, com picos entre 10–20 e 55–65 anos.
     - O peso varia de 20 a 110 kg, concentrando-se principalmente entre 60 e 80 kg.
     - A altura dos pacientes situa-se majoritariamente entre 1,6 e 1,7 metros.
     - O conjunto de dados é predominantemente composto por amostras femininas (aproximadamente 120 de 148) e quase todos os indivíduos são destros.
   - **Resultado:** Após a limpeza, o dataframe passou a conter 137 características.

2. **Construção do Modelo:**
   - Criação de um modelo MLP composto por uma camada de entrada, diversas camadas ocultas com funções de ativação ReLU e camadas de dropout para prevenir o overfitting.
   - A camada de saída é composta por um único neurônio com ativação sigmoide, adequado para a classificação binária.

   Exemplo de código:
   ```python
   def build_mlp(input_shape):
   try:
      model = Sequential()
      model.add(Dense(128, activation="relu", input_shape=input_shape))
      model.add(Dense(64, activation="relu"))
      model.add(Dense(32, activation="relu"))
      model.add(Dense(1, activation="sigmoid"))

      model.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])
        
      model.summary()
   except Exception as e:
      print("Error building model: ", e)

   return model
   ```

3. **Treinamento:**
   - O modelo foi treinado utilizando os dados preparados, com definição adequada de tamanho de lote e número de épocas.
   - Foram utilizados callbacks de early stopping para interromper o treinamento caso a perda de validação não apresentasse melhora.

## Otimização de Hiperparâmetros com Keras Tuner

Para refinar a arquitetura do MLP, o Keras Tuner foi empregado para automatizar a busca pelos hiperparâmetros ideais. Entre os hiperparâmetros explorados estão:
- O número de camadas ocultas e a quantidade de neurônios em cada uma.
- As taxas de dropout, ajustadas para equilibrar o risco de underfitting e overfitting.
- A taxa de aprendizado do otimizador.
- O número de épocas de treinamento.

Exemplo de código:
```python
def build_tuned_mlp(X_train, y_train, X_val, y_val, directory='tuned_models'):
   try:
      def model_builder(hp):
         model = Sequential()
         model.add(Input(shape=(X_train.shape[1],)))

         for i in range(hp.Int("num_layers", 1, 4)):
            model.add(Dense(hp.Int(f"units_{i}", min_value=16, max_value=128, step=32), activation="relu"))
            model.add(Dropout(hp.Float(f"dropout_{i}", 0, 0.15, step=0.05)))

         model.add(Dense(1, activation="sigmoid"))

         model.compile(optimizer=keras.optimizers.Adam(hp.Choice("learning_rate", values=[0.0002, 0.0001, 0.00005])),
                     loss='binary_crossentropy',
                     metrics=['accuracy'])
         return model
      
      tuner = kt.RandomSearch(
         model_builder,
         objective='val_accuracy',
         max_trials=30,
         executions_per_trial=4,
         project_name='mlp_tuning',
         directory=directory
      )

      tuner.search(X_train, y_train, epochs=10, validation_data=(X_val, y_val))
      best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
      model = tuner.hypermodel.build(best_hps)

      return model

   except Exception as e:
      print("Error building model: ", e)
      return None
```

Apesar dessa abordagem, os modelos não conseguiram aprender de forma eficaz os padrões presentes nos dados.

## O Processo

### Importação e Limpeza dos Dados

- Foram importados os dados brutos.
- Realizou-se a exploração das características não relacionadas aos sensores para avaliar correlações.
- Remoção de 12 características com alta intercorrelação, mantendo-se apenas a mais representativa.
- Eliminação de características redundantes (colunas duplicadas de lateralidade e gênero) e do identificador único.
- Realização de análise exploratória utilizando boxplots e histogramas, que revelou:
  - **Idade:** Distribuição ampla, com picos entre 10–20 e 55–65 anos.
  - **Peso:** Varia de 20 a 110 kg, concentrando-se entre 60 e 80 kg.
  - **Altura:** Principalmente entre 1,6 e 1,7 metros.
  - **Gênero:** Predominantemente feminino (cerca de 120 de 148 amostras).
  - **Lateralidade:** Quase todos os indivíduos são destros (com uma exceção).
- **Resultado final:** Dataframe com 137 características.

### Escalonamento e Divisão dos Dados

- Os dados foram escalonados.
- A divisão dos dados foi realizada utilizando amostragem estratificada e agrupada, para levar em conta as múltiplas amostras de cada indivíduo.

### Treinamento e Avaliação do Modelo

- Os resultados iniciais indicaram que o modelo aprendeu apenas uma classe, classificando todas as entradas em uma única categoria.
- Foram testadas diversas abordagens para tratar o desequilíbrio de classes:
  - **Subamostragem:** Não apresentou melhoria.
  - **Seleção de Características com PCA:** Resultados semelhantes foram obtidos.
  - **Ajuste da Função de Perda:** A função de perda foi alterada de entropia cruzada binária para focal loss, a fim de ponderar as classes, mas os resultados permaneceram inalterados.
- A busca por hiperparâmetros com o Keras Tuner também foi aplicada; entretanto, devido à incapacidade do modelo em aprender de forma eficaz, os resultados do ajuste foram inconsistentes.

## Análise e Resultados

Apesar do extenso pré-processamento e de múltiplas estratégias para lidar com o desequilíbrio e a redundância dos dados, os modelos consistentemente falharam em aprender padrões significativos. Todas as abordagens testadas — subamostragem, PCA e funções de perda alternativas — resultaram em modelos que previam apenas uma classe, sugerindo que, com a metodologia atual, não há evidências suficientes para estabelecer uma relação entre os dados do baropodômetro e a classificação da escoliose.

## Conclusão

Neste estágio, não foi possível comprovar nenhuma relação entre o uso das amostras do baropodômetro e a previsão da severidade da escoliose. Embora trabalhos anteriores (como a dissertação de mestrado mencionada) tenham reportado uma acurácia de 75%, esses estudos não levaram em consideração o fato de que múltiplas amostras foram coletadas do mesmo indivíduo. Ao considerar esse fator, nossos modelos não produziram previsões consistentes ou significativas.

## Contribuições

- Tayenne Euqueres
- William de Oliveira Silva

## Contato

Se você tiver alguma dúvida ou comentário, sinta-se à vontade para entrar em contato:  
[GitHub](https://github.com/faduzin) | [LinkedIn](https://www.linkedin.com/in/ericfadul/) | [eric.fadul@gmail.com](mailto:eric.fadul@gmail.com)

## Estrutura do Repositório

```
.
├── assets/           # Contains the images used in the analysis
├── data/             # Stores the datasets required for the project
├── notebooks/        # Includes Jupyter Notebook (.ipynb) files
├── src/              # Package with modules and modularized functions
├── .gitignore        # List of files and directories to be ignored by Git
├── license           # Project's license file
├── readme.eng.md     # English version of the README
└── readme.md         # Main README (in Portuguese), 
```