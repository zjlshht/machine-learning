class Bird():
    def hungry(self):
        self.hungry=True
    def eat(self):
            if self.hungry:
                print('Aaaah ...')
                self.hungry=False
            else:
                print('No,thanks!')