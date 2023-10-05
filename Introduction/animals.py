class Dog:

    def __init__(self, name):
        self.name = name
        self.tricks = []  # empty list of tricks

    def learn_trick(self, trick):
        self.tricks.append(trick)

    def bark(self):
        print(self.name, 'says woof!')


class Cat:

    def __init__(self, name):
        self.name = name
        self.mood = 'purring'

    def meow(self):
        print(self.name, 'meows loud.')
