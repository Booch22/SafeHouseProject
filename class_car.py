class car :
    def __init__( self, color, brand ) :
        self.color = color
        self.brand = brand

    def read( self ) :
        print( self.color )
        print( self.brand )

jd = car( 'red', 'honda' )
jd.read()
print( type(jd) )