import os

os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_path = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_path,'w') as file:
    file.write('NumRooms,Alley,Price\n')
    file.write('NA,Pave,127500\n')
    file.write('2,NA,106000\n')
    file.write('4,NA,178100\n')
    file.write('NA,NA,140000\n')

