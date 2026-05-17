

def airy(l, NA):
    return 0.61 * l / NA


if __name__ == "__main__":
    l0 = 632
    NA0 = 0.7
    r0 = airy(l0, NA0)
    d0 = 2 * r0
    print(f"d0: {d0:.2f} nm")
