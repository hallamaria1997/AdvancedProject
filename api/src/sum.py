import sys

def sum(a,b):
    return a + b

if __name__ == "__main__":
    a = int(sys.argv[1])
    b = int(sys.argv[2])
    print(sum(a, b))