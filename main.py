# from moving import newRemoveMovingObjects
from backgroundReconstruction import iterativeMedianBlending
def main():
    iterativeMedianBlending("images/set1/","set1.jpg")
    iterativeMedianBlending("images/set2/","set2.jpg")
if __name__ == "__main__":
    main()
