@echo off
echo.
echo ========================================
echo    SPIRITUAL STATIC BOTS LAUNCHER
echo    ZeroLight Orbit - 6993 Bots System
echo ========================================
echo.

:MENU
echo Pilih mode peluncuran:
echo.
echo 1. Full Deployment (6993 bots)
echo 2. Interactive Mode
echo 3. Performance Test
echo 4. Demo Mode
echo 5. Monitoring Only
echo 6. Custom Categories
echo 7. Exit
echo.
set /p choice="Masukkan pilihan (1-7): "

if "%choice%"=="1" goto FULL
if "%choice%"=="2" goto INTERACTIVE
if "%choice%"=="3" goto TEST
if "%choice%"=="4" goto DEMO
if "%choice%"=="5" goto MONITOR
if "%choice%"=="6" goto CUSTOM
if "%choice%"=="7" goto EXIT
echo Pilihan tidak valid!
goto MENU

:FULL
echo.
echo Meluncurkan Full Deployment (6993 bots)...
python spiritual-master-launcher.py --mode full
goto END

:INTERACTIVE
echo.
echo Meluncurkan Interactive Mode...
python spiritual-master-launcher.py --mode interactive
goto END

:TEST
echo.
echo Meluncurkan Performance Test...
python spiritual-master-launcher.py --mode test
goto END

:DEMO
echo.
echo Meluncurkan Demo Mode...
python spiritual-master-launcher.py --mode demo
goto END

:MONITOR
echo.
echo Meluncurkan Monitoring Only...
python spiritual-master-launcher.py --mode monitor
goto END

:CUSTOM
echo.
echo Kategori yang tersedia:
echo - ai (AI & Machine Learning)
echo - data (Data Processing & Analytics)
echo - api (API & Integration)
echo - security (Security & Protection)
echo - localization (Localization & Communication)
echo - platform (Platform-Specific)
echo - infrastructure (Infrastructure & Server)
echo.
set /p categories="Masukkan kategori (pisahkan dengan spasi): "
python spiritual-master-launcher.py --mode selective --categories %categories%
goto END

:EXIT
echo.
echo Terima kasih telah menggunakan Spiritual Static Bots!
exit /b 0

:END
echo.
echo Sistem telah selesai berjalan.
pause
goto MENU