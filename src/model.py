# -*- coding: utf-8 -*-
"""
A rede neural (modelo) e o "professor" (treinador online).

O modelo é um MLP simples: 18 entradas -> 64 -> 64 -> 2 saídas (x, y na tela).
A saída passa por uma sigmoide, garantindo que fique sempre em [0, 1] (a tela).

O GazeModel guarda também a memória de exemplos (replay buffer) e o histórico de
loss, para que a tela "Ver parâmetros" possa mostrar o aprendizado acontecendo.
"""
import os
import time
import numpy as np
import torch
import torch.nn as nn

from config import (
    FEATURE_DIM, HIDDEN_DIM, OUTPUT_DIM, LEARNING_RATE, WEIGHT_DECAY,
    BATCH_SIZE, STEPS_PER_FRAME, REPLAY_CAPACITY, MODEL_PATH, DATASET_PATH,
)


class MLP(nn.Module):
    """Perceptron multicamadas: mapeia características do olho -> ponto na tela."""

    def __init__(self, in_dim=FEATURE_DIM, hidden=HIDDEN_DIM, out_dim=OUTPUT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
            nn.Sigmoid(),  # força a saída para a região [0, 1] = a tela
        )

    def forward(self, x):
        return self.net(x)


class GazeModel:
    """Encapsula o modelo, o otimizador, a memória de treino e a persistência."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = MLP().to(self.device)
        self.opt = torch.optim.Adam(
            self.net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        self.loss_fn = nn.MSELoss()

        # Memória de exemplos (replay buffer)
        self.X = np.zeros((REPLAY_CAPACITY, FEATURE_DIM), dtype=np.float32)
        self.Y = np.zeros((REPLAY_CAPACITY, OUTPUT_DIM), dtype=np.float32)
        self.size = 0          # quantas amostras já foram armazenadas
        self.cursor = 0        # posição circular de escrita

        # Estatísticas para a aula
        self.samples_seen = 0
        self.last_loss = float("nan")
        self.loss_history = []  # loss médio por "época" curta, para a curva

    # -- inferência ----------------------------------------------------------
    @torch.no_grad()
    def predict(self, feats):
        """Recebe vetor (FEATURE_DIM,) e devolve (x, y) em [0,1]."""
        x = torch.from_numpy(feats).float().unsqueeze(0).to(self.device)
        y = self.net(x).cpu().numpy()[0]
        return y

    # -- coleta de dados -----------------------------------------------------
    def add_sample(self, feats, target_xy):
        """Guarda um par (características, gabarito) na memória."""
        i = self.cursor
        self.X[i] = feats
        self.Y[i] = target_xy
        self.cursor = (self.cursor + 1) % REPLAY_CAPACITY
        self.size = min(self.size + 1, REPLAY_CAPACITY)
        self.samples_seen += 1

    # -- treino online -------------------------------------------------------
    def train_steps(self, n_steps=STEPS_PER_FRAME, batch_size=BATCH_SIZE):
        """Faz alguns passos de gradiente amostrando da memória.

        É aqui que o modelo "aprende com os próprios erros": comparamos a previsão
        com o gabarito (onde a pessoa realmente estava olhando) e ajustamos os pesos
        para reduzir esse erro.
        """
        if self.size < 8:
            return self.last_loss
        losses = []
        for _ in range(n_steps):
            idx = np.random.randint(0, self.size, size=min(batch_size, self.size))
            xb = torch.from_numpy(self.X[idx]).to(self.device)
            yb = torch.from_numpy(self.Y[idx]).to(self.device)
            pred = self.net(xb)
            loss = self.loss_fn(pred, yb)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            losses.append(loss.item())
        self.last_loss = float(np.mean(losses))
        self.loss_history.append(self.last_loss)
        if len(self.loss_history) > 2000:
            self.loss_history = self.loss_history[-2000:]
        return self.last_loss

    # -- persistência --------------------------------------------------------
    def save(self):
        torch.save({
            "state_dict": self.net.state_dict(),
            "opt_state": self.opt.state_dict(),
            "samples_seen": self.samples_seen,
            "loss_history": self.loss_history[-2000:],
            "feature_dim": FEATURE_DIM,
            "hidden_dim": HIDDEN_DIM,
        }, MODEL_PATH)
        # Salva também o dataset bruto (para inspeção / reuso)
        np.savez_compressed(DATASET_PATH, X=self.X[:self.size], Y=self.Y[:self.size])

    def load(self):
        """Carrega modelo + dataset se existirem. Retorna True se carregou algo."""
        loaded = False
        if os.path.exists(MODEL_PATH):
            ckpt = torch.load(MODEL_PATH, map_location=self.device)
            self.net.load_state_dict(ckpt["state_dict"])
            try:
                self.opt.load_state_dict(ckpt["opt_state"])
            except Exception:
                pass
            self.samples_seen = ckpt.get("samples_seen", 0)
            self.loss_history = ckpt.get("loss_history", [])
            self.last_loss = self.loss_history[-1] if self.loss_history else float("nan")
            loaded = True
        if os.path.exists(DATASET_PATH):
            d = np.load(DATASET_PATH)
            X, Y = d["X"], d["Y"]
            n = min(len(X), REPLAY_CAPACITY)
            self.X[:n] = X[:n]
            self.Y[:n] = Y[:n]
            self.size = n
            self.cursor = n % REPLAY_CAPACITY
            loaded = True
        return loaded

    # -- introspecção para a tela de parâmetros ------------------------------
    def num_parameters(self):
        return sum(p.numel() for p in self.net.parameters())

    def weight_stats(self):
        """Norma L2 de cada camada Linear, para mostrar que os pesos mudam."""
        stats = []
        for name, module in self.net.named_modules():
            if isinstance(module, nn.Linear):
                w = module.weight.detach()
                stats.append((name, float(w.norm().item()), tuple(w.shape)))
        return stats
