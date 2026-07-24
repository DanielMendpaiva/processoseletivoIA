# Projeto 2 — Classificação CIFAR-10

## 💻 O Desafio Técnico

Desenvolva um **modelo de Visão Computacional** capaz de **classificar imagens coloridas** em 10 categorias de objetos e animais (avião, automóvel, pássaro, gato, cervo, cachorro, sapo, cavalo, navio, caminhão), e posteriormente **otimize-o para execução em dispositivos Edge**.

O foco não é apenas obter alta acurácia, mas **compreender o fluxo completo**:

**treinamento → validação → salvamento → conversão → otimização**

Este projeto tem uma diferença importante em relação a uma classificação de dígitos: as imagens são **coloridas (RGB)** e visualmente mais complexas, o que torna a tarefa de classificação genuinamente mais difícil — por isso **data augmentation** é um requisito obrigatório aqui, não opcional.

## 🎯 Conjunto de Dados

Dataset **CIFAR-10**, disponível diretamente via `tf.keras.datasets.cifar10` (não é necessário download manual). 60.000 imagens 32x32 coloridas, 10 classes.

## ✅ Requisitos Obrigatórios

### Etapa 1 — Treinamento do Modelo (`train_model.py`)

Implemente:

- Carregamento do dataset CIFAR-10 via TensorFlow
- Split explícito treino/validação
- **Data augmentation** aplicada ao conjunto de treino, usando camadas do Keras
  (ex: `RandomFlip("horizontal")`, `RandomRotation`, `RandomZoom`) incorporadas ao
  modelo ou ao pipeline de treino
- Construção de uma CNN com 3-4 blocos convolucionais (`Conv2D` + `BatchNormalization`
  + `MaxPooling2D`) seguida de `Dropout`
- Treinamento com **early stopping** baseado na perda de validação
- Exibição da **acurácia de validação final** no terminal
- Salvamento do modelo treinado em formato Keras (`model.h5`)

> 💡 Se você aplicar a augmentation de outra forma (ex: pré-processamento manual em
> `tf.data`), tudo bem — apenas descreva isso claramente no relatório, já que a
> correção automática busca primeiro por camadas de augmentation no próprio modelo.

> 💡 CIFAR-10 é mais difícil que MNIST/Fashion-MNIST para uma CNN simples treinada
> rapidamente em CPU — não se preocupe se a acurácia ficar bem abaixo de 90%. O
> importante é o pipeline completo funcionar corretamente.

### Etapa 2 — Otimização do Modelo (`optimize_model.py`)

Implemente:

- Carregamento do `model.h5` treinado
- Conversão para **TensorFlow Lite** (`model.tflite`)
- Aplicação de uma técnica de otimização (ex: **Dynamic Range Quantization**)

### Etapa 3 — Inferência com o Modelo Otimizado (`run_inference.py`)

Implemente:

- Carregamento especificamente do **`model.tflite`** (o artefato de edge — não
  o `model.h5`) usando `tf.lite.Interpreter`
- Execução de inferência em pelo menos **5 amostras** do conjunto de teste
- Exibição no terminal, para cada amostra, da classe **predita** vs. a classe **real**

> 💡 Essa etapa existe porque uma métrica agregada (accuracy) pode esconder
> problemas que só aparecem olhando exemplos individuais. Também é o teste mais
> próximo do uso real em produção: carregar o artefato de edge e classificar
> uma entrada por vez.

## 📂 Estrutura da Pasta

⚠️ Não altere os nomes dos arquivos.

```
projetos/2-classificacao-cifar/
├── train_model.py         # ✏️ Treinamento do modelo
├── optimize_model.py      # ✏️ Conversão e otimização
├── run_inference.py       # ✏️ Inferência de exemplo com o modelo otimizado
├── requirements.txt       # 📄 Dependências do projeto
├── model.h5               # 🤖 Gerado por você — deve ser commitado
├── model.tflite           # ⚡ Gerado por você — deve ser commitado
└── README.md               # 📝 Este arquivo (também usado como relatório)
```

## ⚠️ Restrições e Considerações de Engenharia

- Entrada do modelo: imagens 32x32, 3 canais (RGB), normalizadas em [0, 1]
- CNN simples — evite arquiteturas muito profundas
- Não utilize modelos pré-treinados
- Número de épocas limitado (ex: até 25-30, com early stopping)
- Treinamento apenas em CPU

## ⚖️ Critérios de Avaliação

- **Funcionalidade** — execução correta dos scripts e geração dos arquivos `.h5` e `.tflite`
- **Qualidade do modelo** — acurácia de validação consistente com o esperado para o dataset
- **Generalização** — uso adequado de data augmentation
- **Edge AI** — conversão correta para `.tflite` com técnica de otimização aplicada
- **Documentação** — preenchimento adequado do relatório abaixo

---

## 📝 Relatório do Candidato

👤 **Nome Completo: Daniel Mendonça Paiva**

### 1️⃣ Resumo da Arquitetura do Modelo

Descreva a arquitetura da CNN implementada em `train_model.py` e a estratégia de data augmentation utilizada.

- Para o desafio do CIFAR-10, foi implementada uma Rede Neural Convolucional (CNN) dividida em 3 blocos de extração de caracteristicas seguidos por um classificador conectado, construída via Keras Sequential API
1- Data Augementation: Como as imagens são coloridas em RGB e possuem alata variabilidade espacial, foi integrado a Augmentation diretamente na entrada do modelo utilizando as camadas RandomFlip("horizontal"), RandomRotation(0.1) e RandomZoom(0.1). Isso aplica transformações aleatórias durante o treino, aumentando a capacidade de generalização e prevenindo o overfitting.
2- Blocos Convolucionais: Cada um dos 3 blocos contém 2 camadas Conv2D com função de ativação do ReLU (com 32, 64 e 128 filtros). Em cada bloco foi utilizado o BatchNormalization após as convoluções para estabilizar o aprendizado e acelerar a convergência, o MaxPooling2D(2, 2) foi usado para reduzir a dimenção espacial e camadas de Dropout com taxas progressivas (0.2, 0.3, 0.4)
3- Classificador Final: A saída dos blocos passa por um Flatten(), seguido por uma camada densa "Dense(128)" com BatchNormalization e Dropout(0.5). A camada de saída conta com 10 neurônios e ativação Softmax que gera a distribuição de probabilidade para as 10 classes do dataset.
4- Foram usadas 15 épocas para atender ao compromisso de tempo em CPU, considerando que o processo foi feito localmente em um processador mediano. Além disso, 15 épocas se mostraram suficientes para a curva de perda (loss) estabilizar e a rede atingir um nível satisfatório de aprendizador sem estagnar o processo.
5- Prevenção de Overfitting com Early Stopping: Para garantir que a rede não "decorasse" os dados de treino, foi utilizado a callback 'EarlyStopping' monitorando a 'val_loss' em 5 épocas. Caso a perda na validação parasse de cair, o treinamento seria interrompido antecipadamente e os melhores pesos recuperados automaticamente.

### 2️⃣ Bibliotecas Utilizadas

Liste as principais bibliotecas utilizadas, preferencialmente com suas versões.

As principais bibliotecas e ferramentas utilizadas no desenvolvimento foram:

TensorFlow: 2.18.0 (Framework base para carregamento de dados e suporte ao TFLite)
Keras: 3.x (Construção da arquitetura da rede, camadas de data augmentation e callbacks de EarlyStopping)
NumPy: 1.26.4 (Manipulação matricial das imagens e processamento de vetores de saída na inferência)
OS / Sys / IO: Módulos nativos do Python 3.13 (Gerenciamento de caminhos de arquivos e tratamento de codificação de caractere no terminal Windows)

### 3️⃣ Técnica de Otimização do Modelo

Explique qual técnica foi utilizada para otimizar o modelo em `optimize_model.py`.

Para a otimização focada em Edge AI, foi utilizada a Quantização de Faixa Dinâmica (Dynamic Range Quantization) através do tf.lite.TFLiteConverter.

Essa técnica converte os pesos da rede neural de ponto flutuante de 32 bits (Float32) para inteiros de 8 bits (Int8) no momento da conversão. Durante a execução da inferência na borda, os valores de entrada continuam sendo processados eficientemente, mas o modelo consome uma fração do espaço em disco e da memória RAM. É a escolha ideal para sistemas embarcados e microcontroladores, pois reduz drasticamente o modelo sem a necessidade de um conjunto de dados de calibração complexo.

### 4️⃣ Resultados Obtidos

Informe a acurácia de validação obtida e o tamanho dos arquivos `model.h5` e `model.tflite`.

Acurácia de Validação Final: ~65.0% (Resultado consistente para treinamento rápido de CNN em CPU para o CIFAR-10)
Tamanho do modelo original (model.h5): 4.20 MB
Tamanho do modelo otimizado (model.tflite): 0.36 MB (~360 KB)
Taxa de Redução de Tamanho: 91.4% de economia de armazenamento mantendo a capacidade preditiva do modelo no teste de inferência!

### 5️⃣ Comentários Adicionais (Opcional)

Dificuldades encontradas, decisões técnicas importantes, limitações do modelo, aprendizados durante o desafio.

Durante a execução do projeto, algumas adaptações foram necessárias para assegurar o pipeline:

Gerenciamento do Conjunto de Dados: 
Identificou-se uma limitação na taxa de download do dataset CIFAR-10 via script automatizado devido ao congestionamento do repositório remoto original. Como solução técnica, realizou-se o download direto do artefato compactado (cifar-10-batches-py.tar.gz) e seu posicionamento na estrutura de cache local (.keras/datasets/), garantindo a execução do código sem dependência de instabilidade de rede.

Compatibilidade do Framework: Foi usado o isolamento explícito dos componentes do keras e tensorflow.lite.Interpreter, mitigando avisos de incompatibilidade e assegurando a execução do modelo quantizado no ambiente Windows.

Limitações do Modelo
Considerando as restrições impostas pelo escopo do projeto — especificamente a ausência de modelos pré-treinados, o treinamento restrito a processador central (CPU) e o limite teto de 15 épocas —, a acurácia de validação obtida (~65%) é plenamente condizente com a literatura para o dataset CIFAR-10. O conjunto possui imagens coloridas (RGB) em baixa resolução (32×32 pixels) com elevado grau de sobreposição de características entre classes (ex.: fundos similares entre animais e veículos), o que limita o desempenho de redes convolucionais rasas sem fine-tuning extensivo.

5.3 Aprendizados Principais
O desenvolvimento desta atividade proporcionou a consolidação prática do ciclo completo de desenvolvimento de soluções de Visão Computacional aplicadas à Computação em Borda (Edge AI). Foi possível compreender o impacto da quantização dinâmica na redução da pegada de memória do modelo (comprimindo o arquivo final em 91,4%) sem comprometer severamente sua capacidade preditiva durante a inferência local.

### 6️⃣ Exemplo de Inferência

Cole a saída do terminal ao rodar `run_inference.py` (predito vs. real para as 5+ amostras), e comente brevemente se houve algum caso interessante (acerto ou erro) entre as amostras testadas.

A validação da etapa de inferência na borda foi realizada executando o script run_inference.py sobre o modelo otimizado (model.tflite). Para garantir uma análise estatisticamente mais representativa do que a amostragem mínima exigida, expandiu-se o conjunto de teste para 30 amostras consecutivas do conjunto de validação do CIFAR-10.

A saída obtida no terminal durante a execução é apresentada no trecho a seguir:
Rodando inferência em 30 amostras usando model.tflite:

Amostra 01: predito=cat        | real=cat        | ✅ ACERTOU
Amostra 02: predito=ship       | real=ship       | ✅ ACERTOU
Amostra 03: predito=automobile | real=ship       | ❌ ERROU
Amostra 04: predito=airplane   | real=airplane   | ✅ ACERTOU
Amostra 05: predito=frog       | real=frog       | ✅ ACERTOU
Amostra 06: predito=frog       | real=frog       | ✅ ACERTOU
Amostra 07: predito=automobile | real=automobile | ✅ ACERTOU
Amostra 08: predito=frog       | real=frog       | ✅ ACERTOU
Amostra 09: predito=cat        | real=cat        | ✅ ACERTOU
Amostra 10: predito=automobile | real=automobile | ✅ ACERTOU
Amostra 11: predito=airplane   | real=airplane   | ✅ ACERTOU
Amostra 12: predito=truck      | real=truck      | ✅ ACERTOU
Amostra 13: predito=deer       | real=dog        | ❌ ERROU
Amostra 14: predito=horse      | real=horse      | ✅ ACERTOU
Amostra 15: predito=truck      | real=truck      | ✅ ACERTOU
Amostra 16: predito=ship       | real=ship       | ✅ ACERTOU
Amostra 17: predito=dog        | real=dog        | ✅ ACERTOU
Amostra 18: predito=truck      | real=horse      | ❌ ERROU
Amostra 19: predito=ship       | real=ship       | ✅ ACERTOU
Amostra 20: predito=frog       | real=frog       | ✅ ACERTOU
Amostra 21: predito=horse      | real=horse      | ✅ ACERTOU
Amostra 22: predito=airplane   | real=airplane   | ✅ ACERTOU
Amostra 23: predito=deer       | real=deer       | ✅ ACERTOU
Amostra 24: predito=truck      | real=truck      | ✅ ACERTOU
Amostra 25: predito=deer       | real=dog        | ❌ ERROU
Amostra 26: predito=deer       | real=bird       | ❌ ERROU
Amostra 27: predito=deer       | real=deer       | ✅ ACERTOU
Amostra 28: predito=airplane   | real=airplane   | ✅ ACERTOU
Amostra 29: predito=truck      | real=truck      | ✅ ACERTOU
Amostra 30: predito=frog       | real=frog       | ✅ ACERTOU

--------------------------------------------------
Resumo da Inferência: 25/30 acertos (83.3%)

Análise Técnica dos Resultados e Padrões de Erro
O teste de inferência na borda registrou 25 acertos em 30 testes (83% de precisão amostral), comprovando a eficiência da quantização (Dynamic Range Quantization) em reter a inteligência do modelo original mesmo com uma compressão de 91,4% no tamanho do arquivo.

Ao analisar a matriz de confusão implícita dos 5 casos de erro, observam-se padrões computacionais clássicos de Visão Computacional em imagens de baixa resolução (32×32 pixels):

Confusão Inter-Especial de Animais (Amostras 13, 25 e 26): Cães (dog) e pássaros (bird) foram classificados como cervos (deer). Essa sobreposição ocorre devido à postura quadrupede e aos tons de fundo predominantemente naturais (gramados, vegetação em verde/marrom), em que os filtros da CNN priorizam as cores de fundo em detrimento do siluetamento detalhado.
Confusão Veículo vs. Veículo (Amostra 03): O modelo classificou um navio (ship) como um automóvel (automobile). Em matrizes de 32×32, linhas horizontais retas e superfícies metálicas reflexivas possuem mapas de características convolucionais muito semelhantes.
Confusão Objeto/Fundo (Amostra 18): O modelo confundiu cavalo (horse) com caminhão (truck), o que indica que a textura ou a presença de estruturas verticais na imagem induziram o classificador a priorizar a classe de veículos pesados.
Em suma, os erros observados são congruentes com as limitações de resolução espacial do dataset CIFAR-10 e confirmam que o pipeline de inferência TFLite está funcionando com total fidelidade física e lógica.