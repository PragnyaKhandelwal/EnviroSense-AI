@echo off
setlocal

set "PIO=%USERPROFILE%\.platformio\penv\Scripts\pio.exe"
if not exist "%PIO%" (
  set "PIO=pio"
)

if "%~1"=="" goto :help

if /I "%~1"=="build" (
  "%PIO%" run -e esp32doit-devkit-v1
  goto :eof
)

if /I "%~1"=="clean" (
  "%PIO%" run -e esp32doit-devkit-v1 -t clean
  "%PIO%" run -e esp32doit-devkit-v1
  goto :eof
)

if /I "%~1"=="upload" (
  if "%~2"=="" (
    "%PIO%" run -e esp32doit-devkit-v1 -t upload
  ) else (
    "%PIO%" run -e esp32doit-devkit-v1 -t upload --upload-port %~2
  )
  goto :eof
)

if /I "%~1"=="monitor" (
  "%PIO%" device monitor -b 115200
  goto :eof
)

if /I "%~1"=="full" (
  if "%~2"=="" (
    "%PIO%" run -e esp32doit-devkit-v1 -t upload
  ) else (
    "%PIO%" run -e esp32doit-devkit-v1 -t upload --upload-port %~2
  )
  if errorlevel 1 goto :eof
  "%PIO%" device monitor -b 115200
  goto :eof
)

:help
echo Usage:
echo   run-pio.bat build
echo   run-pio.bat clean
echo   run-pio.bat upload [COMx]
echo   run-pio.bat monitor
echo   run-pio.bat full [COMx]
echo.
echo Examples:
echo   run-pio.bat upload COM3
echo   run-pio.bat full COM3
