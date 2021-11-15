import tensorflow as tf
import csv
import os
import sys
import numpy as np
import random

ALLOWED_VARIANCE = 9
IDEAL_VARIANCE = 2

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def test_model(valI, model, result):
    temp = list(zip(model.predict(valI), result))
    to_return = list()
    for i in temp:
        predicted = i[0][0]
        actual = i[1][0]
        to_return.append((predicted, actual))
    return to_return


def process_data(filename):
    xtrain = []
    ytrain = []
    with open(filename, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in spamreader:
            ydata = [float(row[1])]
            xdata = [float(row[2]), float(row[3])]
            xtrain.append(np.array(xdata))
            ytrain.append(np.array(ydata))

    train = list(zip(xtrain, ytrain))
    random.shuffle(train)
    x = [x for x, y in train]
    y = [y for x, y in train]

    xtrain = np.array(x[:len(x)//2])
    ytrain = np.array(y[:len(y)//2])

    xval = np.array(x[len(x)//2:])
    yval = np.array(y[len(y)//2:])

    x = np.array(x)
    y = np.array(y)

    return(xtrain, ytrain, xval, yval, x, y)  # Training Input, Training Output, Validation Input, Validation Output, Full Input, Full Output



def test_model_single(inputa, inputb, model):
    print("\n" * 20)
    print(f"{bcolors.BOLD}Predicted Value for above water of {inputa} and below water of {inputb} is:\n{bcolors.OKGREEN}{model.predict(np.array([[float(inputa), float(inputb)]]))[0][0]} inches{bcolors.ENDC} of visibility")


def beautify_output(values):  # (Predicted Value, Actual Value)
    fail_count = 0
    success_count = 0
    twenty_five_bound = int(ALLOWED_VARIANCE * .25)
    within_twenty_five = 0
    fifty_bound = int(ALLOWED_VARIANCE * .5)
    within_fifty = 0
    seventy_five_bound = int(ALLOWED_VARIANCE * .75)
    within_seventy_five = 0
    total_num = len(values)
    for i in values:
        predicted = i[0]
        actual = i[1]
        difference = abs(predicted-actual)
        if difference < seventy_five_bound:
            within_seventy_five += 1
            if difference < fifty_bound:
                within_fifty += 1
                if difference < twenty_five_bound:
                    within_twenty_five += 1
        check = check_difference(difference)
        if check[1] < 0: 
            fail_count += 1
        if check[1] > 0:
            success_count += 1
        print(f"{bcolors.BOLD}{predicted}\t\t{actual}\t\t{check[0]}{difference}{bcolors.ENDC}")
    print("---"*50)
    print(f"Within 25%: {bcolors.BOLD}{bcolors.OKGREEN}{within_twenty_five} {within_twenty_five*100/total_num}% {bcolors.ENDC}Within 50%: {bcolors.BOLD}{bcolors.OKCYAN}{within_fifty} {within_fifty*100/total_num}% {bcolors.ENDC}Within 75%: {bcolors.BOLD}{bcolors.OKBLUE}{within_seventy_five} {within_seventy_five*100/total_num}%{bcolors.ENDC}")
    print(f"{bcolors.BOLD}Fails: {bcolors.FAIL}{fail_count}{bcolors.ENDC} {bcolors.BOLD}Successes:{bcolors.OKGREEN} {success_count}{bcolors.ENDC} {bcolors.BOLD}Fail to Success: {bcolors.OKCYAN}{fail_count/success_count}{bcolors.ENDC} Success to Total: {bcolors.OKCYAN}{success_count/total_num}{bcolors.ENDC}")


def check_difference(value):
    if abs(value) > ALLOWED_VARIANCE:
        return (bcolors.FAIL, -1)
    if abs(value) < IDEAL_VARIANCE:
        return (bcolors.OKGREEN, 1)
    return ("", 0)


def main(datafile, savefile):
    vals = process_data(datafile)
    model = tf.keras.models.load_model(savefile)
    beautify_output(test_model(vals[2], model, vals[3]))


if __name__ == "__main__":
    if len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])  # Datafile Modelfile
    else:
        test_model_single(sys.argv[1], sys.argv[2], tf.keras.models.load_model(sys.argv[3]))  # ValA ValB ModelFile
