@echo off
call C:\ProgramData\Anaconda3\Scripts\activate.bat C:\ProgramData\Anaconda3
call activate old
f:
cd Precipitation-Nowcasting

:loop
python training.py

pause