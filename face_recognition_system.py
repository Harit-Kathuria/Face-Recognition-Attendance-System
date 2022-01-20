#Modules required
import training
import detection

#Global Variables required
images = []
people = []

print('ATTENDANCE RECORDING VIA FACE RECOGNITION SYSTEM')
while True:
    print('1. Record Face\t2. Mark Attendance\t3. Exit\nEnter your choice')
    while True:
        try:
            choice = int(input())
        except ValueError:
            print('Please Enter Integer!!!')
        else:
            break
    if choice == 1:
        encodings, people, images = training.train_system()
    elif choice == 2:
        detection.detect_face(encodings, images,people)
    elif choice == 3:
        break
    else:
        print("Invalid Command")