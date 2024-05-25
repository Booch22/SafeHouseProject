def bi_search( data, target ) :
    l = 0
    r = len( data ) - 1
    found = False

    while l <= r :
        m = ( l + r ) // 2

        if target < data[m] :
            r = m - 1

        elif target > data[m] :
            l = m + 1

        elif target == data[m] :
            print( f'Found {data[m]}' )
            found = True
            break

    if found == False :
        print( f'Not found {target}' )

queue = []

while True :
    print( f'Current queue {queue}' )
    print( 'a : add\nd : delete\ns : sort\nS : search' )
    kb = input( '-> ' )

    if kb == 'a' :
        num = int( input( 'add data in queue : ' ) )
        queue.append( num )

    elif kb == 'd' :
        print( f'delete {queue[0]} from queue' )
        queue.pop( 0 )

    elif kb == 's' :
        print( 'sort data' )
        queue.sort()

    elif kb == 'S' :
        target = int( input( 'Search data : ' ) )
        bi_search( queue, target )


    print( '\n' )
