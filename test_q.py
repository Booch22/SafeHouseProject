def bi_search( data, target ) :
    l = 0
    r = len(data) - 1

    while l <= r :
        m = (l + r) // 2
        
        if target < data[m] :
            r = m - 1

        elif target > data[m] :
            l = m + 1
        
        elif target == data[m] :
            return data[m]


    print( 'Not found' )


queue = []
while True :
    print( f'Current queue {queue}' )
    print( 'a : add\nd : delete\ns : sort\nS : search' )
    kb = input( '-> ' )
    
    if kb == 'a' :
        num = int( input('add data : ') )
        queue.append( num )

    elif kb == 'd' :
        print( f'delete {queue[0]} from queue' )
        queue.pop(0)

    elif kb == 's' :
        print( 'sort data' )
        queue.sort()

    elif kb == 'S' :
        target = int( input( 'Search data -> ' ) )
        data_search = bi_search( queue, target )
        print( f"Found {data_search}" )


    print( '\n' )