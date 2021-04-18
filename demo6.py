class Counterlist(list):
    def __init__(self,*args):
        super().__init__(*args)
        self.counter=0
    def __getitem__(self, index):
        self.counter+=1#访问一次就累加一次
        return super(Counterlist,self).__getitem__(index)
        