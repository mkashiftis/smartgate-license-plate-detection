@echo off
REM SmartGate - double-click to run on Windows
cd /d "%~dp0"

if not exist .venv\Scripts\python.exe (
  echo First run: installing everything. This takes 5-10 minutes and needs internet.
  py -3 -m venv .venv 2>nul || python -m venv .venv
  if not exist .venv\Scripts\python.exe (
    echo Python 3 is not installed. Get it from https://www.python.org/downloads/ and tick "Add python.exe to PATH".
    pause
    exit /b 1
  )
  .venv\Scripts\python -m pip install --upgrade pip
  .venv\Scripts\python -m pip install -r requirements.txt || (echo Install failed - see the message above. & pause & exit /b 1)
  echo Downloading the text-reading model ^(about 100 MB, one time only^)...
  .venv\Scripts\python -c "import easyocr; easyocr.Reader(['en'], gpu=False, verbose=False)"
)

start "" cmd /c "timeout /t 5 >nul & start http://localhost:8000"
.venv\Scripts\python app.py %*
pause
