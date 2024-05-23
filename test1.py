def bi_search( data, target ) :
    l = 0
    r = len( data ) - 1
    m = int( (l + r) / 2 )

    round = 0
    while l <= r :
        print( f"round = {round}" )

        m = int( (l + r) / 2 )
        
        if target < data[m] :
            r = m - 1

        elif target > data[m] :
            l = m + 1

        elif target == data[m] :
            print( 'found target' )
            return data[m]
        
        round += 1

a = [1, 3, 4, 6, 7, 11, 22, 23, 32]
num = bi_search( a, 32 )
print( num )