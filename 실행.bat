@echo off
chcp 65001 >nul
cd /d "%~dp0"

rem 설치가 안 돼 있으면 안내하고 멈춘다 (예전엔 그냥 창이 닫혀서 원인을 알 수 없었다)
if not exist ".venv\Scripts\pythonw.exe" (
    echo.
    echo   [설치가 아직 안 됐습니다]
    echo.
    echo   처음 받은 폴더라면 setup.bat 을 먼저 실행해 주세요.
    echo   setup.bat 을 더블클릭하면 필요한 것들이 자동으로 깔립니다.
    echo.
    pause
    exit /b 1
)

rem 패키지가 빠져 있으면 pythonw 는 아무 말 없이 죽는다. 미리 확인한다.
".venv\Scripts\python.exe" -c "import cv2, numpy, mss, PIL, tkinter" 2>nul
if errorlevel 1 (
    echo.
    echo   [필요한 패키지가 빠져 있습니다]
    echo.
    echo   setup.bat 을 다시 실행해 주세요. 무엇이 빠졌는지는 아래에 나옵니다.
    echo.
    ".venv\Scripts\python.exe" -c "import cv2, numpy, mss, PIL, tkinter"
    echo.
    pause
    exit /b 1
)

start "" ".venv\Scripts\pythonw.exe" main.py
