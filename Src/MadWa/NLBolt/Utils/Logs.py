from datetime import datetime

r'''
A simple tool to make logs
'''


__all__ = ['Log']


class Log:
    def __init__(self, fname=None, erase=True):
        self.fname = fname
        if not fname is None:
            if erase:
                open(fname, 'w').close()

    def write(self, stri):
        if self.fname is None:
            print(stri)
        else:
            with open(self.fname, 'a') as f:
                f.write(stri + '\n')

    def Twrite(self, stri):
        dt = datetime.now()
        dtstr = dt.strftime("%m/%d/%Y, %H:%M:%S.%f")
        striAll = dtstr + ' : ' + stri
        if self.fname is None:
            print(striAll)
        else:
            with open(self.fname, 'a') as f:
                f.write(striAll + '\n') 
            
    def cut(self):
        if self.fname is None:
            print('----------')
        else:
            with open(self.fname, 'a') as f:
                f.write('---------- \n') 

    def Twrite2(self, stri):
        dt = datetime.now()
        if self.fname is None:
            print(dt)
            print(stri)
        else:
            with open(self.fname, 'a') as f:
                f.write('\n') 
                f.write(dt.strftime("%m/%d/%Y, %H:%M:%S.%f")+'\n')
                f.write(stri + '\n') 