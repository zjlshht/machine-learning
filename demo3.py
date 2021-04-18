class Bird:
    def hungry(self):
        self.hungry=True
    def eat(self):
            if self.hungry:
                print('Aaaah ...')
                self.hungry=False
            else:
                print('No,thanks!')
class SongBird(Bird):
    def sound(self):
        super().hungry()
        self.sound='Squawk' 
    def sing(self):
            print(self.sound)