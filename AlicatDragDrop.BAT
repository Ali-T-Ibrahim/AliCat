:loop
   python "C:\Users\Ali--\Desktop\Machine Learning Pytorch\Alicat\alicatModel.py" %*
   GetKey /N
if %errorlevel% == 0 goto loop
echo The key pressed have this code: %errorlevel%python

