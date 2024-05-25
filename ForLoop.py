import random

rand_data = [ random.randint(1, 10) for _ in range(5) ]
print( f'Current data is {rand_data}\n' )

for i in range( len(rand_data) - 1 ) :
    print( f'index      = {rand_data[i]}' )
    print( f'next index = {rand_data[i+1]}')
    print( '\n' )