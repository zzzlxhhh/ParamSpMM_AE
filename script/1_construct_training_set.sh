conda activate param_env
cd ..
# run PCSR.py with default augments
# It may take hours to finish for all matrices and all dim
# The defualt  '--dim_list' is  '[16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256]'
python PCSR.py
cd ./script
