@echo off
call C:\ProgramData\Anaconda3\Scripts\activate.bat C:\ProgramData\Anaconda3
call activate project
f:
cd Precipitation-Nowcasting

:loop
python main.py
goto loop

pause