@echo off
chcp 65001 > nul
echo ========================================================
echo   AI Data Insight Builder 실행 중...
echo   브라우저에서 http://127.0.0.1:5000 으로 접속하세요.
echo   종료하려면 이 창에서 Ctrl + C 를 누르세요.
echo ========================================================
call .\venv\Scripts\activate.bat
python app.py
pause
