import numpy as np

class Animal:
    def __init__(self):
        self.a = 0;
        self.b = 1;
        print('Base init.')

    def fun_A(self):
        print('a ',self.a)
        print('Base fun_A.')

    def fun_B(self):
        print('b ',self.b)
        print('Base fun_B.')

# 函数重载会覆盖原来的函数，包括构造函数，因此原基类构造函数里的变量都没有
# 但是加上super(Cat, self).__init__()，就会执行基类的构造函数了
class Cat(Animal):
    def __init__(self):
        super(Cat, self).__init__()
        self.a = 10;
        # self.b = 11;
        self.c = 1;
        print('Cat init.')

    def fun_A(self):
        print('Cat fun_A.')

if __name__ == '__main__':
    cat = Cat()
    cat.fun_A()
    cat.fun_B()
