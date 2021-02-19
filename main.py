#!/usr/local/bin/python3.8

#######################################
## author : Aung Khant Myo(TwizzyIndy)
## date : Feb/2021
#######################################


from imageai.Detection import ObjectDetection
import os

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
    execution_path = os.getcwd()

    detector = ObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(os.path.join(execution_path, "yolo.h5"))
    detector.loadModel()

    custom_objects = detector.CustomObjects(person=True)
    detections = detector.detectCustomObjectsFromImage(custom_objects=custom_objects,
     input_image=imagePath,
      output_image_path=outputImagePath,
      minimum_percentage_probability=10)

    for eachObject in detections:
        print(eachObject["name"] , " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"] )
        print("--------------------------------")

    print("\n\nFound person in image : " + str(len(detections)))

    return

def helpInfos():
    print("")
    print("Count People")
    print("TwizzyIndy (Feb/2021)")
    print("python3 main.py -input test1.jpg")
    print("")
    return

if __name__ == "__main__":
    main()