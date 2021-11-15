import csv
import sys
import numpy as np
import tensorflow as tf
from sklearn.linear_model import LinearRegression
from sklearn import svm
import random
import pickle


ALLOWED_VARIANCE = 6
IDEAL_VARIANCE = 2
EXCLUDED_RANGE = (0, 0)


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


def make_model(value_list):
    xtrain, ytrain, xval, yval, x, y, savefile = value_list
    model = tf.keras.models.Sequential(
        [
            tf.keras.Input(shape=(2,)),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1)
        ]
    )

    print(model.summary())

    model.compile(loss="mean_squared_error", optimizer="adam", metrics=["mean_squared_error"])

    model.fit(xtrain, ytrain, epochs=1000, batch_size=1, validation_data=(xval, yval))
    model.save(savefile)

    return model


def reconstruct_model(model_location):
    return tf.keras.models.load_model(model_location)


def test_model(valI, model, result):
    temp = list(zip(model.predict(valI), result))
    to_return = list()
    for i in temp:
        predicted = i[0][0]
        actual = i[1][0]
        to_return.append((predicted, actual))
    return to_return


def check_difference(value):
    if abs(value) > ALLOWED_VARIANCE:
        return (bcolors.FAIL, -1)
    if abs(value) < IDEAL_VARIANCE:
        return (bcolors.OKGREEN, 1)
    return ("", 0)


def make_regression(type, xvals, yvals):
    if type == "SVM":
        reg = svm.SVR()
    elif type == "LR":
        reg = LinearRegression()
    return reg.fit(xvals, yvals)


def test_regression(regression, xvals, yvals):
    newx = regression.predict(xvals)
    linked_vals = list(zip(newx, list(yvals)))
    to_return = list()
    for i in linked_vals:
        to_return.append((i[0], i[1][0]))
    return to_return


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


def main(datafile, savefile, task=None):
    vals = process_data(datafile)
    if task is None:
        to_pass = list(vals) + [savefile]
        model = make_model(to_pass)
        beautify_output(test_model(vals[4], model, vals[5]))
    elif task == "Load":
        model = reconstruct_model(savefile)
        beautify_output(test_model(vals[4], model, vals[5]))
    elif task == "Linear":
        regression_model = make_regression("LR", vals[4], vals[5])
    elif task == "SVM":
        regression_model = make_regression("SVM", vals[4], vals[5])
    if task == "SVM" or task == "Linear":
        beautify_output(test_regression(regression_model, vals[4], vals[5]))


if __name__ == "__main__":
    if len(sys.argv) > 3:
        main(sys.argv[1], sys.argv[2], sys.argv[3])  # Datafile Savefile None,Linear,SVM,Load 
    else:
        main(sys.argv[1], sys.argv[2])  # Datafile Savefile