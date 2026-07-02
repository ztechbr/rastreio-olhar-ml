# Rastreio do Olhar — Demonstração de Aprendizado de Máquina

Uma aplicação didática para mostrar, na prática, **como uma máquina aprende**.

A câmera observa seus olhos e uma **bola tenta seguir para onde você está olhando**.
O sistema **aprende com os próprios erros**: ele compara para onde *colocou* a bola
com para onde você *realmente* estava olhando e vai ajustando os pesos da rede neural
para melhorar o sincronismo. Em paralelo, ele **gera os dados de treinamento** e
**salva o modelo** em disco.

Feito para uso em sala de aula: os três botões da tela inicial deixam claras as três
fases de um projeto de ML — **treinar**, **testar** e **inspecionar o modelo**.

---

## Como funciona (a intuição da aula)

1. **Extração de características (features).** O MediaPipe FaceMesh encontra 478 pontos
   no rosto, incluindo a íris. A partir deles montamos um vetor de **18 números** que
   descreve a direção do olhar (posição da íris dentro do olho, abertura dos olhos,
   posição e rotação aproximada da cabeça). Veja `src/tracking.py`.

2. **O modelo.** Uma rede neural simples (MLP): `18 → 64 → 64 → 2`. A saída são as
   coordenadas `(x, y)` na tela, em `[0, 1]`. Veja `src/model.py`.

3. **Aprender com os próprios erros.** No modo **Treinar**, a bola **verde** é o
   *gabarito* (você é instruído a olhar para ela). A bola **vermelha** é o *palpite*
   do modelo. A cada quadro medimos o erro (distância entre as duas) e damos passos de
   gradiente para reduzi-lo — você **vê a bola vermelha grudar na verde** conforme o
   modelo aprende. Esse é o coração do ML: minimizar o erro entre previsão e gabarito.

4. **Dados + modelo salvos.** Cada exemplo `(features → posição real)` é guardado; o
   dataset (`data/dataset.npz`) e o modelo (`models/model.pt`) são salvos
   automaticamente durante o treino.

---

## Tela inicial

- **1 · Treinar modelo** — olhe fixo para a bola verde; a vermelha aprende a te seguir.
- **2 · Testar modelo** — só a bola vermelha, seguindo livremente o seu olhar.
- **3 · Ver parâmetros** — arquitetura, nº de parâmetros, amostras coletadas, loss,
  learning rate, normas dos pesos e a **curva de aprendizado** (loss ao longo do tempo).

Atalhos: `ESC` volta/sai · `Q` sai · `F` tela cheia · `C` liga/desliga a câmera.

---

## Requisitos

- **Windows** com webcam.
- **Python 3.10** (o 3.12+/3.14 ainda não tem wheels de MediaPipe/PyTorch — use o 3.10).
- **GPU NVIDIA opcional** (ex.: RTX 4050). O PyTorch é instalado com **CUDA 12.4**, que
  roda nessa placa. Sem GPU, ele usa a CPU automaticamente.

## Instalação

```bat
setup.bat
```

Isso cria um ambiente virtual em `.venv`, instala o PyTorch com CUDA e as demais
dependências, e mostra se a GPU foi reconhecida.

## Executar

```bat
run.bat
```

Dica de sala: no modo **Treinar**, faça ~1–2 minutos seguindo a bola verde com os
olhos (movimente a cabeça o mínimo possível). Depois vá em **Testar** e veja o
resultado; volte a **Treinar** sempre que quiser melhorar.

---

## Estrutura

```
rastreio-olhar-ml/
├── src/
│   ├── main.py       # interface (Pygame) + máquina de estados das telas
│   ├── tracking.py   # webcam + MediaPipe -> vetor de 18 características
│   ├── model.py      # rede neural (MLP), treino online e salvamento
│   └── config.py     # caminhos e hiperparâmetros (mexa aqui na aula!)
├── data/             # dataset gerado (dataset.npz)
├── models/           # modelo salvo (model.pt)
├── requirements.txt
├── setup.bat
└── run.bat
```

## Ideias para a aula

- Mude o `LEARNING_RATE` em `src/config.py` e mostre o efeito na curva de loss.
- Compare treinar por 20 s vs. 2 min: quanto o erro em pixels cai?
- Apague `models/model.pt` e `data/dataset.npz` para começar do zero na frente da turma.
- Abra `src/model.py` e mostre onde exatamente o "erro" vira ajuste dos pesos
  (`loss.backward()` + `opt.step()`).

---

*Projeto educacional. O rastreamento de olhar por webcam é aproximado — o objetivo é
ensinar o ciclo de aprendizado de máquina, não fazer eye-tracking de precisão.*
