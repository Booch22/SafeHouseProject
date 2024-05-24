def bi_search( data, target ) :
    l = 0
    r = len( data ) - 1
    found = False

    while l <= r :
        m = ( l + r ) // 2
        print( f"l = {l}, r = {r}, m = {m}")

        if target > data[m] :
            l = m + 1

        elif target < data[m] :
            r = m - 1

        elif target == data[m] :
            print( f'Found {target}' )
            found = True
            break
           
    if found == False :
        print( f'Not found {target}' )

queue = []

while True :
    print( f'Current queue {queue}' )
    print( 'a : add\nd : delete\ns : sort\nS : search' )
    kb = input( '=> ' )
    
    try :
        if kb == 'a' :
            num = int( input( 'add data to queue : ' ) )
            queue.append( num )

        elif kb == 'd' :
            print( f'delete {queue[0]} from queue ' )
            queue.pop(0)

        elif kb == 's' :
            print( f'sort queue' )
            queue.sort()

        elif kb == 'S' :
            target = int( input('search target : ') )
            bi_search( queue, target )

    except :
        print( 'try again' )

    
    print( '\n' )