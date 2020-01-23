import math


def sigmoid_(x):
    if isinstance(x, list):
        return list(map(sigmoid_, x))
    elif isinstance(x, int) or isinstance(x, float):
        return 1.0 / (1 + math.exp(-x))
    else:
        None


# x = -4
# print(sigmoid_(x))
# x = 2
# print(sigmoid_(x))
# x = [-4, 2, 0]
# print(sigmoid_(x))
