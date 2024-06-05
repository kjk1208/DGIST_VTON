
class A():
    def __init__(self):
        self.warp_flow_blks = []
        self.encode_output_chs = [
            320,
            320,
            640,
            640,
            640,    #1280
            1280,   #1280
            1280,   #
            1280, 
            1280
            ]
                
        self.encode_output_chs2 = [
            320,
            320,
            640,
            640,
            640,    #1280
            1280,   #1280
            1280,   #
            1280, 
            1280
        ]
        
        for idx, (in_ch, cont_ch) in enumerate(zip(self.encode_output_chs, self.encode_output_chs2)):
            print(f"idx : {idx}")
            print(f"in_ch : {in_ch}")
            print(f"incont_ch_ch : {cont_ch}")
            self.warp_flow_blks.append((in_ch, cont_ch))
            
        a = list(reversed(self.warp_flow_blks))
        print(a)
            
            
if __name__ == "__main__":
    A()
            