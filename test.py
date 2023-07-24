class Rectangle:
    def __init__(self, length, width):
        self.length = length
        self.width = width

    def area(self):
        return self.length * self.width

    def perimeter(self):
        return 2 * self.length + 2 * self.width

# Here we declare that the Square class inherits from the Rectangle class
class Square(Rectangle):
    pass
    #def __init__(self, length):
    #    super().__init__(length, length)

if __name__ == "__main__":
    sq = Square(4,5)
    print(dir(Rectangle))