@echo off
setlocal EnableDelayedExpansion
title Factor Terminal - launcher

rem ---------------------------------------------------------------------------
rem Starts the API and the web app, waits until both answer, then opens the
rem browser. Safe to run twice: anything already serving is left alone rather
rem than started a second time.
rem ---------------------------------------------------------------------------

cd /d "%~dp0"

set "API_PORT=8100"
set "WEB_PORT=3100"
set "PY=.venv\Scripts\python.exe"
set "API_URL=http://127.0.0.1:%API_PORT%/api/health"
set "WEB_URL=http://127.0.0.1:%WEB_PORT%/"
set "OPEN_URL=http://localhost:%WEB_PORT%"

echo.
echo   FACTOR TERMINAL
echo   ---------------
echo.

rem --- prerequisites ---------------------------------------------------------

if not exist "%PY%" (
    echo   [X] No virtual environment found at %PY%
    echo.
    echo       Create it first:
    echo         python -m venv .venv
    echo         .venv\Scripts\pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

where npm >nul 2>&1
if errorlevel 1 (
    echo   [X] npm is not on PATH. Install Node.js, then run this again.
    echo.
    pause
    exit /b 1
)

where curl.exe >nul 2>&1
if errorlevel 1 (
    echo   [X] curl.exe not found. It ships with Windows 10 1803 and later.
    echo.
    pause
    exit /b 1
)

rem --- backend ---------------------------------------------------------------

call :is_up "%API_URL%"
if "!UP!"=="1" (
    echo   [=] API already serving on port %API_PORT%
) else (
    echo   [^>] Starting API on port %API_PORT% ...
    start "Factor Terminal - API" /min cmd /c ^
        ""%PY%" -m uvicorn backend.app.main:app --host 127.0.0.1 --port %API_PORT%"
)

rem --- frontend --------------------------------------------------------------

call :is_up "%WEB_URL%"
if "!UP!"=="1" (
    echo   [=] Web app already serving on port %WEB_PORT%
    goto :wait_for_both
)

if not exist "frontend\node_modules" (
    echo   [^>] Installing web dependencies, this takes a minute on first run ...
    pushd frontend
    call npm install --no-audit --no-fund
    if errorlevel 1 (
        popd
        echo   [X] npm install failed.
        pause
        exit /b 1
    )
    popd
)

if not exist "frontend\.next\BUILD_ID" (
    echo   [^>] Building the web app, this takes a minute on first run ...
    pushd frontend
    call npm run build
    if errorlevel 1 (
        popd
        echo   [X] Build failed. Run "npm run build" in the frontend folder to see why.
        pause
        exit /b 1
    )
    popd
)

echo   [^>] Starting web app on port %WEB_PORT% ...
start "Factor Terminal - Web" /min cmd /c "cd /d "%~dp0frontend" && npm run start"

rem --- wait for both to answer -----------------------------------------------

:wait_for_both
echo.
echo   Waiting for the app to come up ...

set /a TRIES=0
:poll
set /a TRIES+=1

call :is_up "%API_URL%"
set "API_UP=!UP!"
call :is_up "%WEB_URL%"
set "WEB_UP=!UP!"

if "!API_UP!"=="1" if "!WEB_UP!"=="1" goto :ready

if !TRIES! GEQ 60 (
    echo.
    echo   [X] Timed out after about 60 seconds.
    if not "!API_UP!"=="1" echo       The API never answered - check the "Factor Terminal - API" window.
    if not "!WEB_UP!"=="1" echo       The web app never answered - check the "Factor Terminal - Web" window.
    echo.
    pause
    exit /b 1
)

call :sleep 1
goto :poll

rem --- open the browser ------------------------------------------------------

:ready
echo   [OK] Both services are up.
echo.
start "" "%OPEN_URL%"

echo   Opened %OPEN_URL%
echo.
echo   API docs   http://127.0.0.1:%API_PORT%/docs
echo   Logs       the two minimised "Factor Terminal" windows
echo   To stop    close those two windows
echo.
call :sleep 6
exit /b 0

rem ---------------------------------------------------------------------------
rem :sleep <seconds>
rem
rem ping rather than timeout: timeout aborts with "input redirection is not
rem supported" whenever stdin is redirected, which would turn the poll loop into a
rem busy spin that exhausts its retries in well under a second.
rem ---------------------------------------------------------------------------
:sleep
set /a _PINGS=%~1+1
ping -n !_PINGS! 127.0.0.1 >nul 2>&1
exit /b 0

rem ---------------------------------------------------------------------------
rem :is_up <url>  ->  UP=1 when the URL answers, else UP=0
rem
rem An HTTP probe rather than netstat, for two reasons. netstat prints its state
rem column in the system language - "ABHOEREN" on a German Windows, not
rem "LISTENING" - so matching on that word silently never fires. And a socket can
rem be bound while the server is still booting and not yet serving, which is
rem exactly the window this loop is waiting through.
rem ---------------------------------------------------------------------------
:is_up
set "UP=0"
curl.exe -s -o nul --max-time 3 %~1 >nul 2>&1
if not errorlevel 1 set "UP=1"
exit /b 0
