import os
import sys


# Function to rename multiple files
def main(foldername, depth_in_inches, date_time):
    for count, filename in enumerate(os.listdir(foldername)):
        dst = str(count) + "-" + depth_in_inches + "-" + date_time + ".png"
        src = foldername + filename
        dst = foldername + dst

        # rename() function will
        # rename all the files
        os.rename(src, dst)


# Driver Code
if __name__ == '__main__':
    # Calling main() function
    main(sys.argv[1], sys.argv[2], sys.argv[3])
