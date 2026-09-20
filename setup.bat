@echo off
cd /d "%~dp0"
py -m venv .venv
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
echo.
echo 설치가 끝났습니다. 실행.bat 으로 프로그램을 여세요.
pause
