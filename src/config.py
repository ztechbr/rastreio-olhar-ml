# -*- coding: utf-8 -*-
"""
Configurações centrais do projeto.

Este arquivo concentra caminhos, hiperparâmetros e constantes para que a aula
possa alterar UM lugar só e ver o efeito no aprendizado (ex.: learning rate).
"""
import os

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

DATASET_PATH = os.path.join(DATA_DIR, "dataset.npz")   # dados de treino gerados
MODEL_PATH = os.path.join(MODELS_DIR, "model.pt")       # modelo salvo

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Modelo / aprendizado
# ---------------------------------------------------------------------------
FEATURE_DIM = 18        # tamanho do vetor de características extraído do rosto
HIDDEN_DIM = 64         # neurônios por camada escondida
OUTPUT_DIM = 2          # (x, y) normalizados na tela, em [0, 1]

LEARNING_RATE = 1e-3    # taxa de aprendizado do Adam
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 64         # amostras por passo de treino
STEPS_PER_FRAME = 3     # quantos passos de gradiente por quadro (treino online)
REPLAY_CAPACITY = 30000 # memória de amostras para treino (replay buffer)

AUTOSAVE_SECONDS = 5.0  # de quanto em quanto tempo salvamos modelo + dados

# ---------------------------------------------------------------------------
# Câmera
# ---------------------------------------------------------------------------
CAM_INDEX = 0
CAM_WIDTH = 640
CAM_HEIGHT = 480

# ---------------------------------------------------------------------------
# Janela / UI
# ---------------------------------------------------------------------------
WIN_WIDTH = 1280
WIN_HEIGHT = 720
FPS = 60

# Suavização (EMA) da bola prevista para reduzir tremor visual
PRED_SMOOTHING = 0.35   # 0 = sem suavização, ~1 = muito lento

# Cores (R, G, B)
COL_BG = (18, 18, 24)
COL_TEXT = (230, 230, 235)
COL_MUTED = (140, 140, 155)
COL_TARGET = (60, 220, 120)     # bola verde (para onde você DEVE olhar = gabarito)
COL_PRED = (240, 80, 90)        # bola vermelha (palpite do modelo)
COL_ACCENT = (90, 160, 255)
COL_ERROR_LINE = (250, 200, 60)
