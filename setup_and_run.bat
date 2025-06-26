@echo off
echo Creating Python virtual environment...
python -m venv venv
call venv\Scripts\activate

echo Installing required libraries...
pip install --upgrade pip
pip install tensorflow numpy matplotlib seaborn scikit-learn

echo Running CNN Object Recognition script...
python cifar10_cnn.py

pause