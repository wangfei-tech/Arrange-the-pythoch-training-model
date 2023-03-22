class Person(object):
    def __init__(self, name, age):
        self.name = name
        self.age = age


class Male(Person):
    def __init__(self, name, age, height):
        super().__init__(name, age)
        self.height = height
        print(super())

if __name__ == '__main__':
    m = Male('xiaoming', 18, 172)
    print(m.__dict__)