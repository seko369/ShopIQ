import csv
import sys
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.1

def main():
    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])  #feachors and labels (evidence,labels)

    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    y_pred = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, y_pred)

    # Print results
    print(f"Correct: {(y_test == y_pred).sum()}")
    print(f"Incorrect: {(y_test != y_pred).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    labels = []
    evidences = []
    str_key = {        
        "Jan": 0,
        "Feb": 1,
        "Mar": 2,
        "Apr": 3,
        "May": 4,
        "June": 5,
        "Jul": 6,
        "Aug": 7,
        "Sep": 8,
        "Oct": 9,
        "Nov": 10,
        "Dec": 11,
        "Returning_Visitor":1,
        "New_Visitor": 0,
        "Other": 0,
        "TRUE":1,
        "FALSE":0  
        }
    # list_str = ["feb","mar","may","nov","Returning_Visitor","TRUE","FALSE",]
    with open(filename) as file:
        list_lines = file.readlines()[1:]


    for l in list_lines :
        mini_evidenc = []
        spliter = l.strip().split(",")
        for ar in spliter :
            if ar in str_key :
                mini_evidenc.append(str_key[ar])
            
            elif str(float(ar)) == ar or "." in ar:
                        mini_evidenc.append(float(ar))
            elif str(int(ar)) == ar : 
                        mini_evidenc.append(int(ar))
            else:
                print(f"push the true data for {ar}")
                    

        a = mini_evidenc.pop(-1)
        evidences.append(mini_evidenc)
        labels.append(a)
    data_set = (evidences,labels)
    return data_set


def train_model(evidence, labels):

    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence,labels)
    return model



def evaluate(y_test, y_pred):
    list_index = []
    for num,val in enumerate(y_test):
         if val == 1 :
              list_index.append(num)
    how_much_1 = len(list_index)
    counter_pred = 0
    for i in list_index :
         if y_pred[i] == 1 :
              counter_pred+=1
    sensivity = (counter_pred/how_much_1)

    list_index1 = []
    for num,val in enumerate(y_test):
         if val == 0 :
              list_index1.append(num)
    how_much_12 = len(list_index1)
    counter_pred1 = 0
    for i in list_index1 :
         if y_pred[i] == 0 :
              counter_pred1+=1
    specifi = (counter_pred1/how_much_12)

    return (sensivity,specifi)

         

if __name__ == "__main__":
    main()
