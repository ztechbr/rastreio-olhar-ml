@echo off
REM ============================================================================
REM  Configuracao do ambiente (Windows)
REM  Cria um ambiente virtual com Python 3.10 e instala tudo, incluindo
REM  o PyTorch com CUDA 12.4 (roda na sua RTX 4050).
REM ============================================================================
setlocal

echo.
echo [1/4] Criando ambiente virtual com Python 3.10...
py -3.10 -m venv .venv
if errorlevel 1 (
  echo ERRO: Python 3.10 nao encontrado. Instale-o em python.org e tente de novo.
  exit /b 1
)

call .venv\Scripts\activate.bat

echo.
echo [2/4] Atualizando o pip...
python -m pip install --upgrade pip

echo.
echo [3/4] Instalando PyTorch com CUDA 12.4 (aceleracao na GPU NVIDIA)...
pip install torch --index-url https://download.pytorch.org/whl/cu124

echo.
echo [4/4] Instalando as demais dependencias...
pip install -r requirements.txt

echo.
echo ============================================================
echo  Pronto! Verificando se a GPU foi reconhecida pelo PyTorch:
python -c "import torch; print('CUDA disponivel:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
echo ============================================================
echo  Agora rode:  run.bat
echo.
endlocal
