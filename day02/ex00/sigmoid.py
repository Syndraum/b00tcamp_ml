import math


def sigmoid_(x):
    def func(x):
        return (1.0 / (1 + math.exp(-x)))
    if isinstance(x, list):
        mylist = []
        for elmt in x:
            mylist.append(func(elmt))
        return mylist
    elif isinstance(x, int):
        return func(x)
    else:
        return None


# x = -4
# print(sigmoid_(x))
# x = 2
# print(sigmoid_(x))
# x = [-4, 2, 0]
# print(sigmoid_(x))
