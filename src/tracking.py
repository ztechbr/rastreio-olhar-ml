# -*- coding: utf-8 -*-
"""
Rastreamento facial e extração de características (features) do olhar.

Usa o MediaPipe FaceMesh (com refine_landmarks=True, que adiciona os pontos da
íris) para transformar a imagem da webcam em um vetor numérico de 18 números que
descreve PARA ONDE a pessoa está olhando (posição da íris dentro do olho, abertura
dos olhos, posição e rotação aproximada da cabeça).

Esse vetor é a "entrada" da rede neural. Repare que NÃO dizemos ao modelo a regra
de como o olho vira em direção à tela — ele aprende isso sozinho a partir dos
exemplos coletados no modo Treinar.
"""
import numpy as np
import cv2
import mediapipe as mp

from config import FEATURE_DIM, CAM_INDEX, CAM_WIDTH, CAM_HEIGHT

# Índices de landmarks do FaceMesh que nos interessam ------------------------
# Íris (só existem com refine_landmarks=True)
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]
# Cantos dos olhos
L_EYE_OUT, L_EYE_IN = 33, 133      # olho esquerdo (na imagem)
R_EYE_OUT, R_EYE_IN = 263, 362     # olho direito
# Pálpebras (cima/baixo) para abertura
L_LID_TOP, L_LID_BOT = 159, 145
R_LID_TOP, R_LID_BOT = 386, 374
# Referências do rosto (bounding box e pose)
NOSE = 1
FACE_TOP, FACE_BOT = 10, 152
CHEEK_L, CHEEK_R = 234, 454

EPS = 1e-6


class GazeTracker:
    """Captura a webcam e produz o vetor de características do olhar."""

    def __init__(self):
        self.cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,   # habilita landmarks da íris
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.last_frame_rgb = None   # último frame (RGB) para exibir na tela

    def read(self):
        """Lê um quadro, roda o FaceMesh e devolve (features, frame_rgb).

        features: np.ndarray shape (FEATURE_DIM,) ou None se nenhum rosto.
        frame_rgb: imagem RGB (espelhada) para mostrar no preview da câmera.
        """
        ok, frame = self.cap.read()
        if not ok:
            return None, self.last_frame_rgb

        frame = cv2.flip(frame, 1)  # espelha (efeito "espelho")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.last_frame_rgb = rgb

        result = self.face_mesh.process(rgb)
        if not result.multi_face_landmarks:
            return None, rgb

        lm = result.multi_face_landmarks[0].landmark
        feats = self._extract_features(lm)
        return feats, rgb

    # -- extração de características ------------------------------------------
    def _extract_features(self, lm):
        def p(i):
            return np.array([lm[i].x, lm[i].y], dtype=np.float32)

        def mean_pts(idxs):
            return np.mean([[lm[i].x, lm[i].y] for i in idxs], axis=0).astype(np.float32)

        # Bounding box do rosto (para normalizar e ser invariante a distância)
        pts = np.array([[lm[i].x, lm[i].y] for i in (FACE_TOP, FACE_BOT, CHEEK_L, CHEEK_R, NOSE)])
        minx, miny = pts.min(axis=0)
        maxx, maxy = pts.max(axis=0)
        w = max(maxx - minx, EPS)
        h = max(maxy - miny, EPS)

        def norm(pt):
            return np.array([(pt[0] - minx) / w, (pt[1] - miny) / h], dtype=np.float32)

        l_iris = mean_pts(LEFT_IRIS)
        r_iris = mean_pts(RIGHT_IRIS)

        l_out, l_in = p(L_EYE_OUT), p(L_EYE_IN)
        r_out, r_in = p(R_EYE_OUT), p(R_EYE_IN)
        l_top, l_bot = p(L_LID_TOP), p(L_LID_BOT)
        r_top, r_bot = p(R_LID_TOP), p(R_LID_BOT)
        nose = p(NOSE)

        # Razão da íris DENTRO do olho -> sinal forte de direção do olhar
        l_ratio_x = (l_iris[0] - l_out[0]) / ((l_in[0] - l_out[0]) + EPS)
        l_ratio_y = (l_iris[1] - l_top[1]) / ((l_bot[1] - l_top[1]) + EPS)
        r_ratio_x = (r_iris[0] - r_out[0]) / ((r_in[0] - r_out[0]) + EPS)
        r_ratio_y = (r_iris[1] - r_top[1]) / ((r_bot[1] - r_top[1]) + EPS)

        # Abertura dos olhos (normalizada pela largura do rosto)
        l_open = (l_bot[1] - l_top[1]) / w
        r_open = (r_bot[1] - r_top[1]) / w

        # Pose aproximada da cabeça
        yaw_proxy = (abs(nose[0] - lm[CHEEK_L].x) - abs(lm[CHEEK_R].x - nose[0])) / w
        pitch_proxy = (abs(nose[1] - lm[FACE_TOP].y) - abs(lm[FACE_BOT].y - nose[1])) / h

        l_iris_n = norm(l_iris)
        r_iris_n = norm(r_iris)
        nose_n = norm(nose)

        feats = np.array([
            l_iris_n[0], l_iris_n[1],
            r_iris_n[0], r_iris_n[1],
            l_ratio_x, l_ratio_y,
            r_ratio_x, r_ratio_y,
            l_open, r_open,
            nose_n[0], nose_n[1],
            minx + w / 2.0, miny + h / 2.0,   # centro do rosto (translação)
            w, h,                             # tamanho do rosto (distância)
            yaw_proxy, pitch_proxy,
        ], dtype=np.float32)

        assert feats.shape[0] == FEATURE_DIM, feats.shape
        # Sanidade: troca NaN/inf por 0
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        return feats

    def release(self):
        try:
            self.cap.release()
        except Exception:
            pass
        try:
            self.face_mesh.close()
        except Exception:
            pass
