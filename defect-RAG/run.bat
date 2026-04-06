@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8

REM Set Python path
set PYTHON="C:\Users\WYG6SZH\.conda\envs\langflowPython310\python.exe"

REM Run Streamlit app
%PYTHON% -m streamlit run main.py

pause
