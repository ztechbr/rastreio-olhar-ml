# -*- coding: utf-8 -*-
"""
Aplicativo principal — interface (Pygame) e máquina de estados.

Telas:
  MENU     -> Treinar modelo / Testar modelo / Ver parâmetros
  TREINAR  -> bola VERDE (alvo/gabarito) + bola VERMELHA (previsão). O modelo
              aprende ao vivo comparando as duas. Dados e modelo são salvos.
  TESTAR   -> só a bola VERMELHA seguindo o seu olhar (sem gabarito).
  PARAMS   -> mostra arquitetura, nº de parâmetros, amostras, loss e a curva.

Atalhos globais: ESC = voltar ao menu | Q = sair | F = tela cheia | C = câmera.
"""
import math
import time
import numpy as np
import pygame

from config import (
    WIN_WIDTH, WIN_HEIGHT, FPS, PRED_SMOOTHING, AUTOSAVE_SECONDS,
    COL_BG, COL_TEXT, COL_MUTED, COL_TARGET, COL_PRED, COL_ACCENT, COL_ERROR_LINE,
)
from tracking import GazeTracker
from model import GazeModel

# Estados
MENU, TREINAR, TESTAR, PARAMS = "MENU", "TREINAR", "TESTAR", "PARAMS"


class TargetMover:
    """Move a bola-alvo (verde) suavemente entre pontos aleatórios da tela.

    Ficar parado alguns instantes ("dwell") em cada ponto ajuda a coletar
    exemplos limpos; o movimento suave entre pontos cobre a tela inteira.
    """

    def __init__(self, w, h):
        self.w, self.h = w, h
        self.pos = np.array([0.5, 0.5], dtype=np.float32)   # normalizado [0,1]
        self.src = self.pos.copy()
        self.dst = self._rand()
        self.t = 0.0
        self.move_time = 1.6
        self.dwell_time = 0.7
        self.phase = "move"

    def _rand(self):
        return np.array([np.random.uniform(0.07, 0.93),
                         np.random.uniform(0.09, 0.91)], dtype=np.float32)

    def update(self, dt):
        self.t += dt
        if self.phase == "move":
            k = min(self.t / self.move_time, 1.0)
            s = 0.5 - 0.5 * math.cos(math.pi * k)   # easing (suave nas pontas)
            self.pos = self.src + (self.dst - self.src) * s
            if k >= 1.0:
                self.phase, self.t = "dwell", 0.0
        else:
            if self.t >= self.dwell_time:
                self.src = self.pos.copy()
                self.dst = self._rand()
                self.phase, self.t = "move", 0.0
        return self.pos

    def px(self):
        return int(self.pos[0] * self.w), int(self.pos[1] * self.h)


class App:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Rastreio do Olhar — Aprendizado de Máquina")
        self.w, self.h = WIN_WIDTH, WIN_HEIGHT
        self.fullscreen = False
        self.screen = pygame.display.set_mode((self.w, self.h), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Segoe UI", 22)
        self.font_big = pygame.font.SysFont("Segoe UI", 40, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 18)

        self.state = MENU
        self.show_cam = True
        self.running = True

        # Componentes de ML
        self.tracker = GazeTracker()
        self.model = GazeModel()
        loaded = self.model.load()
        self.status = ("Modelo carregado." if loaded else "Modelo novo (sem treino ainda).")

        self.mover = TargetMover(self.w, self.h)
        self.pred_smooth = np.array([0.5, 0.5], dtype=np.float32)
        self.have_face = False
        self.last_autosave = time.time()

        # Botões do menu (rótulo, estado destino, tecla)
        self.menu_items = [
            ("Treinar modelo", TREINAR, "1"),
            ("Testar modelo", TESTAR, "2"),
            ("Ver parametros", PARAMS, "3"),
        ]
        self.menu_rects = []

    # -- utilidades ----------------------------------------------------------
    def resize(self, w, h):
        self.w, self.h = max(w, 640), max(h, 480)
        self.mover.w, self.mover.h = self.w, self.h
        if not self.fullscreen:
            self.screen = pygame.display.set_mode((self.w, self.h), pygame.RESIZABLE)

    def toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        if self.fullscreen:
            info = pygame.display.Info()
            self.w, self.h = info.current_w, info.current_h
            self.mover.w, self.mover.h = self.w, self.h
            self.screen = pygame.display.set_mode((self.w, self.h), pygame.FULLSCREEN)
        else:
            self.w, self.h = WIN_WIDTH, WIN_HEIGHT
            self.mover.w, self.mover.h = self.w, self.h
            self.screen = pygame.display.set_mode((self.w, self.h), pygame.RESIZABLE)

    def text(self, s, x, y, font=None, color=COL_TEXT, center=False):
        font = font or self.font
        surf = font.render(s, True, color)
        rect = surf.get_rect()
        if center:
            rect.center = (x, y)
        else:
            rect.topleft = (x, y)
        self.screen.blit(surf, rect)
        return rect

    def draw_camera_pip(self, frame_rgb):
        if frame_rgb is None or not self.show_cam:
            return
        pw, ph = 240, 180
        surf = pygame.surfarray.make_surface(np.swapaxes(frame_rgb, 0, 1))
        surf = pygame.transform.smoothscale(surf, (pw, ph))
        x, y = self.w - pw - 16, 16
        self.screen.blit(surf, (x, y))
        pygame.draw.rect(self.screen, COL_ACCENT, (x, y, pw, ph), 2)
        dot = (60, 220, 120) if self.have_face else (240, 80, 90)
        pygame.draw.circle(self.screen, dot, (x + 12, y + ph - 12), 6)

    # -- loop principal ------------------------------------------------------
    def run(self):
        while self.running:
            dt = self.clock.tick(FPS) / 1000.0
            self.handle_events()

            # Lê a câmera uma vez por quadro (usado por todos os estados)
            feats, frame_rgb = self.tracker.read()
            self.have_face = feats is not None

            self.screen.fill(COL_BG)
            if self.state == MENU:
                self.render_menu()
            elif self.state == TREINAR:
                self.update_train(dt, feats)
            elif self.state == TESTAR:
                self.update_test(feats)
            elif self.state == PARAMS:
                self.render_params()

            if self.state in (TREINAR, TESTAR):
                self.draw_camera_pip(frame_rgb)
            self.draw_footer()
            pygame.display.flip()

            self.maybe_autosave()

        self.shutdown()

    def handle_events(self):
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                self.running = False
            elif e.type == pygame.VIDEORESIZE and not self.fullscreen:
                self.resize(e.w, e.h)
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_q:
                    self.running = False
                elif e.key == pygame.K_ESCAPE:
                    if self.state != MENU:
                        self.model.save()
                        self.state = MENU
                    else:
                        self.running = False
                elif e.key == pygame.K_f:
                    self.toggle_fullscreen()
                elif e.key == pygame.K_c:
                    self.show_cam = not self.show_cam
                elif self.state == MENU:
                    if e.unicode == "1":
                        self.state = TREINAR
                    elif e.unicode == "2":
                        self.state = TESTAR
                    elif e.unicode == "3":
                        self.state = PARAMS
            elif e.type == pygame.MOUSEBUTTONDOWN and self.state == MENU:
                for rect, target in self.menu_rects:
                    if rect.collidepoint(e.pos):
                        self.state = target

    # -- TELA: MENU ----------------------------------------------------------
    def render_menu(self):
        self.text("Rastreio do Olhar", self.w // 2, 90, self.font_big, COL_TEXT, center=True)
        self.text("Demonstracao de Aprendizado de Maquina — a bola aprende a seguir seu olhar",
                  self.w // 2, 135, self.font, COL_MUTED, center=True)

        self.menu_rects = []
        bw, bh, gap = 460, 74, 22
        x = self.w // 2 - bw // 2
        y0 = 220
        mouse = pygame.mouse.get_pos()
        for i, (label, target, key) in enumerate(self.menu_items):
            rect = pygame.Rect(x, y0 + i * (bh + gap), bw, bh)
            hover = rect.collidepoint(mouse)
            col = COL_ACCENT if hover else (40, 40, 55)
            pygame.draw.rect(self.screen, col, rect, border_radius=12)
            pygame.draw.rect(self.screen, COL_ACCENT, rect, 2, border_radius=12)
            self.text(f"[{key}]  {label}", rect.centerx, rect.centery,
                      self.font, COL_TEXT, center=True)
            self.menu_rects.append((rect, target))

        self.text(self.status, self.w // 2, y0 + 3 * (bh + gap) + 20,
                  self.font, COL_MUTED, center=True)
        dev = str(self.model.device).upper()
        self.text(f"Dispositivo de treino: {dev}   |   Amostras coletadas: {self.model.size}",
                  self.w // 2, y0 + 3 * (bh + gap) + 55, self.font_small, COL_MUTED, center=True)

    # -- TELA: TREINAR -------------------------------------------------------
    def update_train(self, dt, feats):
        target = self.mover.update(dt)          # gabarito (normalizado [0,1])
        tx, ty = self.mover.px()

        if feats is not None:
            # 1) guarda o exemplo (features -> onde a pessoa DEVE estar olhando)
            self.model.add_sample(feats, target)
            # 2) aprende com os proprios erros (passos de gradiente)
            self.model.train_steps()
            # 3) mostra o palpite atual do modelo
            pred = self.model.predict(feats)
            self.pred_smooth = (PRED_SMOOTHING * self.pred_smooth
                                + (1 - PRED_SMOOTHING) * pred)

        px = int(self.pred_smooth[0] * self.w)
        py = int(self.pred_smooth[1] * self.h)

        # linha de erro entre gabarito e previsao
        if feats is not None:
            pygame.draw.line(self.screen, COL_ERROR_LINE, (tx, ty), (px, py), 2)
        # bola verde (alvo) e vermelha (previsao)
        pygame.draw.circle(self.screen, COL_TARGET, (tx, ty), 26)
        pygame.draw.circle(self.screen, (255, 255, 255), (tx, ty), 26, 2)
        pygame.draw.circle(self.screen, COL_PRED, (px, py), 18)

        # HUD
        err_px = math.hypot(tx - px, ty - py) if feats is not None else float("nan")
        self.text("TREINAR — olhe FIXO para a bola verde; a vermelha vai aprendendo a te seguir",
                  20, 16, self.font, COL_TEXT)
        loss = self.model.last_loss
        loss_s = f"{loss:.5f}" if loss == loss else "--"
        self.text(f"Loss: {loss_s}    Erro atual: {err_px:5.0f}px    Amostras: {self.model.size}",
                  20, 48, self.font_small, COL_ACCENT)
        if not self.have_face:
            self.text("Rosto nao detectado — ajuste a iluminacao / posicao",
                      self.w // 2, self.h - 60, self.font, COL_PRED, center=True)

    # -- TELA: TESTAR --------------------------------------------------------
    def update_test(self, feats):
        if feats is not None:
            pred = self.model.predict(feats)
            self.pred_smooth = (PRED_SMOOTHING * self.pred_smooth
                                + (1 - PRED_SMOOTHING) * pred)
        px = int(self.pred_smooth[0] * self.w)
        py = int(self.pred_smooth[1] * self.h)

        # rastro suave
        pygame.draw.circle(self.screen, COL_PRED, (px, py), 22)
        pygame.draw.circle(self.screen, (255, 255, 255), (px, py), 22, 2)

        self.text("TESTAR — a bola vermelha segue para onde voce olha (sem gabarito)",
                  20, 16, self.font, COL_TEXT)
        self.text(f"Amostras treinadas: {self.model.size}    "
                  f"Loss final: {self.model.last_loss:.5f}" if self.model.last_loss == self.model.last_loss
                  else "Ainda sem treino — use 'Treinar modelo' primeiro",
                  20, 48, self.font_small, COL_ACCENT)
        if not self.have_face:
            self.text("Rosto nao detectado — ajuste a iluminacao / posicao",
                      self.w // 2, self.h - 60, self.font, COL_PRED, center=True)

    # -- TELA: PARAMS --------------------------------------------------------
    def render_params(self):
        self.text("Parametros do Modelo", 40, 30, self.font_big, COL_TEXT)
        x = 40
        y = 100
        lines = [
            f"Dispositivo (device)........: {str(self.model.device).upper()}",
            f"Arquitetura.................: MLP {self.model.net.net[0].in_features}"
            f" -> 64 -> 64 -> 2  (saida sigmoide)",
            f"Total de parametros.........: {self.model.num_parameters():,}",
            f"Amostras na memoria.........: {self.model.size:,}",
            f"Amostras vistas (total).....: {self.model.samples_seen:,}",
            f"Loss atual (MSE)............: "
            + (f"{self.model.last_loss:.6f}" if self.model.last_loss == self.model.last_loss else "--"),
            f"Learning rate...............: {self.model.opt.param_groups[0]['lr']}",
        ]
        for ln in lines:
            self.text(ln, x, y, self.font_small, COL_TEXT)
            y += 30

        y += 10
        self.text("Norma dos pesos por camada (muda conforme aprende):", x, y, self.font_small, COL_MUTED)
        y += 28
        for name, norm, shape in self.model.weight_stats():
            self.text(f"  {name:<10} shape={str(shape):<12} |W|={norm:.3f}",
                      x, y, self.font_small, COL_TEXT)
            y += 26

        # Curva de loss
        self.draw_loss_curve(x, y + 10, 560, 180)

    def draw_loss_curve(self, x, y, w, h):
        self.text("Curva de aprendizado (loss ao longo do treino):",
                  x, y - 26, self.font_small, COL_MUTED)
        pygame.draw.rect(self.screen, (30, 30, 42), (x, y, w, h), border_radius=8)
        pygame.draw.rect(self.screen, COL_MUTED, (x, y, w, h), 1, border_radius=8)
        hist = self.model.loss_history
        if len(hist) < 2:
            self.text("(treine o modelo para ver a curva)", x + 16, y + h // 2 - 10,
                      self.font_small, COL_MUTED)
            return
        data = hist[-w:] if len(hist) > w else hist
        lo, hi = min(data), max(data)
        rng = (hi - lo) or 1e-6
        n = len(data)
        pts = []
        for i, v in enumerate(data):
            px = x + int(i / (n - 1) * (w - 8)) + 4
            py = y + h - 8 - int((v - lo) / rng * (h - 16))
            pts.append((px, py))
        pygame.draw.lines(self.screen, COL_TARGET, False, pts, 2)
        self.text(f"max={hi:.4f}", x + w - 120, y + 6, self.font_small, COL_MUTED)
        self.text(f"min={lo:.4f}", x + w - 120, y + h - 24, self.font_small, COL_MUTED)

    # -- rodape / persistencia ----------------------------------------------
    def draw_footer(self):
        hint = "ESC voltar/sair   |   Q sair   |   F tela cheia   |   C camera"
        self.text(hint, 20, self.h - 30, self.font_small, COL_MUTED)

    def maybe_autosave(self):
        if self.state == TREINAR and time.time() - self.last_autosave > AUTOSAVE_SECONDS:
            self.model.save()
            self.last_autosave = time.time()

    def shutdown(self):
        try:
            self.model.save()
        except Exception:
            pass
        self.tracker.release()
        pygame.quit()


if __name__ == "__main__":
    App().run()
