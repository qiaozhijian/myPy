import cv2
import os
from PIL import Image
# dataset = 'hactl'
dataset = 'taipocity'
path = '/media/qzj/Windows/love/code/VCP/data/ECCV/visualization/{}_visualization/'.format(dataset)
path_pcl_day_image = path+'day/image/'
path_pcl_night_image = path+'night/image/'
path_pcl_day = path+'day/pcl/'
path_pcl_night = path+'night/pcl/'
path_pcl_none = path+'none/'
path_pcl_dcp = path+'dcp/'
path_pcl_icp = path+'icp/'
path_pcl_our = path+'our/'
path_pcl_our2 = path+'our2/'
path_video = path+'video/'

def jpg2video(sp, fps):
    """ 将图片合成视频. sp: 视频路径，fps: 帧率 """
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    images = os.listdir('mv')
    im = Image.open('mv/' + images[0])
    vw = cv2.VideoWriter(sp, fourcc, fps, im.size)

    os.chdir('mv')
    for image in range(len(images)):
        # Image.open(str(image)+'.jpg').convert("RGB").save(str(image)+'.jpg')
        jpgfile = str(image + 1) + '.jpg'
        try:
            frame = cv2.imread(jpgfile)
            vw.write(frame)
        except Exception as exc:
            print(jpgfile, exc)
    vw.release()
    print(sp, 'Synthetic success!')

def video(path,file,image=False):
    path = path
    videoPath = path_video + file+'.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 25
    if image:
        im = Image.open(path + str(100000) + '.png')
        size = (512,288)
    else:
        im = Image.open(path + str(0) + '.png')
        size = im.size
    videoWriter = cv2.VideoWriter(videoPath, fourcc, fps, size)
    files = os.listdir(path)
    frames = len(files)
    for i in range(0, frames):
        if image:
            i = i + 100000
        # print(i)
        framePath = path + str(i) + '.png'
        frame = cv2.imread(framePath)
        frame = cv2.resize(frame, size)
        videoWriter.write(frame)
    videoWriter.release()
    print('over')

if __name__ == '__main__':

    # video(path_pcl_night_image,'night_image',image=True)
    # video(path_pcl_day_image,'day_image',image=True)
    video(path_pcl_none,'pcl_none')
    video(path_pcl_day,'pcl_day')
    video(path_pcl_night,'pcl_night')
    # video(path_pcl_dcp,'dcp')
    # video(path_pcl_our,'our')
    # video(path_pcl_our2,'our2')