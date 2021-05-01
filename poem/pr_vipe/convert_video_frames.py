class KeyPoint:
    name = ""
    x = 0
    y = 0
    score = 0
    def __init__(self,name,x,y,score):
        self.x = x
        self.y = y
        self.name = name
        self.score = score


class VideoResponse:
    width = 0
    height = 0
    frames = []
    def __init__(self,width,height):
        self.width = width
        self.height = height

class FrameResponse:
    points = []
    def __init__(self,points):
        self.points = points

joints = ["nose","leftShoulder","rightShoulder","leftElbow","rightElbow","leftWrist","rightWrist", "leftHip","rightHip","leftKnee","rightKnee","leftAnkle","rightAnkle"]

def convert(video_response:VideoResponse):
    list_for_poem = [video_response.width, video_response.height]
    for frame in video_response.frames:
        jointDict = {}
        for point in frame.points:
            jointDict[point.name] = [point.x, point.y, point.score]
        for joint in joints:
            list_for_poem += jointDict[joint]

    return list_for_poem


            
    
