@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo.
echo   미니맵 감시 알리미 - 설치
echo   ----------------------------------------
echo.

rem 1. 파이썬이 있는지
py -V >nul 2>&1
if errorlevel 1 (
    echo   [파이썬이 없습니다]
    echo.
    echo   https://www.python.org/downloads/ 에서 받아 설치해 주세요.
    echo   설치할 때 "Add python.exe to PATH" 를 꼭 체크하세요.
    echo   설치가 끝나면 이 창을 닫고 setup.bat 을 다시 실행하면 됩니다.
    echo.
    pause
    exit /b 1
)
for /f "delims=" %%v in ('py -V 2^>^&1') do echo   파이썬 확인: %%v

rem 2. 가상환경
if not exist ".venv\Scripts\python.exe" (
    echo   가상환경 만드는 중...
    py -m venv .venv
    if errorlevel 1 (
        echo.
        echo   [가상환경을 만들지 못했습니다]
        echo   폴더 쓰기 권한이 없거나 경로에 문제가 있을 수 있습니다.
        echo.
        pause
        exit /b 1
    )
) else (
    echo   가상환경 확인: 이미 있음
)

rem 3. 패키지
echo   패키지 설치 중... (처음 한 번은 몇 분 걸립니다)
".venv\Scripts\python.exe" -m pip install --upgrade pip --quiet
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo   [패키지 설치에 실패했습니다]
    echo   인터넷 연결을 확인하고 다시 실행해 주세요.
    echo.
    pause
    exit /b 1
)

rem 4. 진짜로 불러와지는지 확인
".venv\Scripts\python.exe" -c "import cv2, numpy, mss, PIL, tkinter"
if errorlevel 1 (
    echo.
    echo   [설치는 됐지만 불러오지 못하는 게 있습니다]
    echo   위 메시지를 확인해 주세요.
    echo.
    pause
    exit /b 1
)

echo.
echo   ----------------------------------------
echo   설치가 끝났습니다. 실행.bat 으로 프로그램을 여세요.
echo.
pause
