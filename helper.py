def writeToFile(filename, value):
    f = open(filename, "a")
    f.write(str(value) + "\n")


def deleteDataFromFile(filename):
    f = open(filename, "r+")
    f.truncate(0)
    f.close()
