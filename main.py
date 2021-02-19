#!/usr/local/bin/python3.8

#######################################
## author : Aung Khant Myo(TwizzyIndy)
## date : Feb/2021
#######################################


from imageai.Detection import ObjectDetection
import os
import warnings

# reduce log level for tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def main():
    
    if len(os.sys.argv) < 2:
        helpInfos()
        return
    
    imagePath = ""

    if os.sys.argv[1] == "-input":
        imagePath = os.sys.argv[2]
    else:
        helpInfos()
        return
    
    outputImagePath = os.path.basename(imagePath) + "_out" + os.path.splitext(imagePath)[1]
    count_people_in_image(imagePath, outputImagePath)

    return

def helpInfos():
    print("")
    print("Count People")
    print("TwizzyIndy (Feb/2021)")
    print("python3 main.py -input test1.jpg")
    print("")
    return

def count_people_in_image(inputImage, outputImage):
    execution_path = os.getcwd()

    detector = ObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(os.path.join(execution_path, "yolo.h5"))
    detector.loadModel()

    custom_objects = detector.CustomObjects(person=True)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        detections = detector.detectCustomObjectsFromImage(custom_objects=custom_objects,
                                                        input_image=inputImage,
                                                        output_image_path=outputImage,
                                                        minimum_percentage_probability=10) # for detecting more people, you can reduce probability number here

    for eachObject in detections:
        print(eachObject["name"] , " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"] )
        print("--------------------------------")

    print("\n\nFound person(s) in image : " + str(len(detections)))
    return

if __name__ == "__main__":
    main()