# -*- coding: utf-8 -*-

__author__ = 'Ziqin (Shaun) Rong'
__maintainer__ = 'Ziqin (Shaun) Rong'
__email__ = 'rongzq08@gmail.com'


class A(object):

    def __init__(self, n):
        self.n = n


if __name__ == "__main__":
    a1 = A(1)
    a2 = A(2)

    print(a1.class_var)
    print(a2.class_var)

    a1.class_var = 3
    print(a1.class_var)
    print(a2.class_var)
