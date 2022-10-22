import cv2
import numpy as np
from numpy import linalg as LA
from numpy.lib.stride_tricks import broadcast_arrays
import datetime
import csv
from operator import itemgetter


from flask import Flask, request, redirect, render_template, flash

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def starmap_serch():
  img_dir = "static/imgs/"
  if request.method == 'GET': img_path=None
  elif request.method == 'POST':
    
    stream = request.files['img'].stream
    img_array = np.asarray(bytearray(stream.read()), dtype=np.uint8)
    img = cv2.imdecode(img_array, 1) 

    #画像保存
    dt_now = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
    img_origin = img_dir + dt_now + ".jpg"
    cv2.imwrite(img_origin, img)
    #処理開始
    imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(imgray,127,255,0)
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    h, w = imgray.shape

    dis0 = 10000
    rng = 0
    fg = 0
    orion = []
    fg2 = 0
    stars = []
    stars_c = []
    size = []

    ori=0
    cyg=0
    tau=0
    can=0
    ari=0
    gem=0

    sum_c=0
    w_h= w/80
    h_h=h/80
    diser = (w_h*w_h)+(h_h*h_h)

    orion_d=np.array([
      [0.40,-50],#bell
      [0.95,29],#saip
      [0.54,12],#alni
      [0.52,4],#alla
      [0.52,-4]#mintaka
    ])

    cyg_d=np.array([
      [1.31,85],#Fawaris
      [2.35,68],#lwing1
      [2.75,68],#lwing2
      [1.22,-109],#Aljanah
      [2.28,-96],#rwing
      [1.20,166],#Azel
      [2.53,169],#Albireo
    ])

    tau_d=np.array([
      [0.92,-28],
      [0.12,148],
      [0.23,143],#hia
      [0.53,156],
      [1.1,152],
      [0.85,96],#pre
      #[0.53,93],
      #[0.38,66],
      [0.2,74],#ain
      [0.2,111]
    ])

    can_d=np.array([
      [0.42,142],
      [0.36,90],
      [0.21,68],
      [0.66,16],
      [0.86,15],#
      [0.96,20],
      [1.21,20],
      [0.94,5],
      [0.60,2],
      [1.23,-10],
      [1.13,-35],
      [0.24,-51],
      [0.52,-34],
      [0.44,-89]#mil
    ])

    aries_d=np.array([
      [2.65,163],
      [1.35,10]
    ])

    gemeni_d=np.array([
      [0.56,81],
      [0.87,145],
      [1.87,103],
      [2.83,118],
      [4.56,103],
      [2.61,94],
      [4.26,92],
      [1.0,59],
      [1.74,40],
      [2.83,28],
      [3.04,67],
      [4.13,79],
      [4.22,71],
      [4.57,69],
      [5.2,66]
    ])

    for cnt in contours:
      M = cv2.moments(cnt)
      if M['m00'] >0.0 :
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        csize = M['m00']
        rng =rng + 1
        stars.append(list([cx,cy,csize]))
        #cv2.circle(img,(cx,cy), 10,(255,0,0),0)

    stars = sorted(stars, key=itemgetter(2), reverse=True)

    def angle(a,b):
      dot = a @ b
      cos = dot / (np.linalg.norm(a) * np.linalg.norm(b))
      rad = np.arccos(cos)
      return np.rad2deg(rad)

    def scoring(data,Alpha,Beta,DISORI):

      Alpha_a=[Alpha[0]-Beta[0],Alpha[1]-Beta[1]]
      Alpha_l=np.linalg.norm(Alpha_a)
      axis = np.rad2deg(np.arctan2(Alpha_a[0],Alpha_a[1]))
      #cv2.line(img,(Alpha[0],Alpha[1]),(Beta[0],Beta[1]),(255,255,255), thickness=1)
      if(Alpha_l*Alpha_l < diser*DISORI):
        return 0,0,0
      sum_c=0
      sum_quan=0
      map_score=0
      starmap=[]
      for data_s in data:
        map_p,sum_c=autos(Alpha[0],Alpha[1],data_s[0],Alpha_l,data_s[1],axis,sum_c)
        map_score = map_score+map_p[2]  
        starmap.append(map_p)
        sum_quan = sum_quan +1
      

      if(map_score/sum_c > 100 and sum_c > sum_quan*2/3):

        starmap.append([Alpha[0],Alpha[1]])
        starmap.append([Beta[0],Beta[1]])
        map_score = map_score + (Alpha[2]*2)+ (Beta[2]*2)
        return starmap,map_score,1

      #逆

      Beta_a=[Beta[0]-Alpha[0],Beta[1]-Alpha[1]]
      Beta_l=np.linalg.norm(Beta_a)
      axis_b = np.rad2deg(np.arctan2(Beta_a[0],Beta_a[1]))
      sum_c=0
      sum_quan=0
      map_score=0
      starmap=[]
      for data_s in data:
        map_p,sum_c=autos(Beta[0],Beta[1],data_s[0],Beta_l,data_s[1],axis_b,sum_c)
        map_score = map_score+map_p[2]  
        starmap.append(map_p)
        sum_quan = sum_quan +1
      

      if(map_score/sum_c > 100 and sum_c > sum_quan*2/3):
        starmap.append([Beta[0],Beta[1]])
        starmap.append([Alpha[0],Alpha[1]])
        map_score = map_score + (Alpha[2]*2)+ (Beta[2]*2)
        return starmap,map_score,1

      return 0,0,0

    def autos(x, y, r, l, th, axis,sum_c):
      theta =  axis - th + 180
      r1 = r * l

      x1 = r1 * np.sin(np.deg2rad(theta)) + x
      y1 = r1 * np.cos(np.deg2rad(theta)) + y
      x2 = round(x1)
      y2 = round(y1)
      if 0 < x2 < w and 0 < y2 < h:
        auto_b=[x2,y2]
        score=-500
        for auto in stars_c:
          _dis_x = auto[0] - x2
          _dis_y = auto[1] - y2
          _dis = _dis_x**2 + _dis_y**2
          if _dis < diser:
            score_b =diser - _dis + (auto[2]*2)
            if (score_b > score):
              score = score_b
              auto_b=[auto[0],auto[1]]
        return np.array([auto_b[0],auto_b[1],score],dtype='int32'),sum_c+1

      return np.array([x2,y2,0]),sum_c

    if rng > 50:
      for k in range(50):
        stars_c.append(stars[k])
    else:
      stars_c = stars


    for Beta in stars_c:
      for Alpha in stars_c:
        if (Alpha == Beta):
          break
        if(Beta[2]+Alpha[2] < 10):
          break
        else:

          if(ori != 1):
            orion,orion_score,ori = scoring(orion_d,Alpha,Beta,4)
          if(cyg != 1):
            cygnus,cyg_score,cyg = scoring(cyg_d,Alpha,Beta,2)
          if(tau != 1):
            taurus,tau_score,tau = scoring(tau_d,Alpha,Beta,4)
          if(can != 1):
            cannis,cannis_score,can = scoring(can_d,Alpha,Beta,4)
          if(ari != 1):
            aries,aries_score,ari = scoring(aries_d,Alpha,Beta,2)
          if(gem != 1):
            gemeni,gemeni_score,gem = scoring(gemeni_d,Alpha,Beta,4)
          else:
            break
      if(ori==1 or cyg == 1 or tau==1 or can==1 or ari ==1 or gem ==1):
        break


    if ori == 1:
      cv2.line(img,(orion[5][0],orion[5][1]),(orion[2][0],orion[2][1]),(255,255,255), thickness=1)
      cv2.line(img,(orion[2][0],orion[2][1]),(orion[1][0],orion[1][1]),(255,255,255), thickness=1)
      cv2.line(img,(orion[1][0],orion[1][1]),(orion[6][0],orion[6][1]),(255,255,255), thickness=1)
      cv2.line(img,(orion[6][0],orion[6][1]),(orion[4][0],orion[4][1]),(255,255,255), thickness=1)
      cv2.line(img,(orion[4][0],orion[4][1]),(orion[0][0],orion[0][1]),(255,255,255), thickness=1)
      cv2.line(img,(orion[5][0],orion[5][1]),(orion[0][0],orion[0][1]),(255,255,255), thickness=1)
      cv2.line(img,(orion[2][0],orion[2][1]),(orion[3][0],orion[3][1]),(255,255,255), thickness=1)
      cv2.line(img,(orion[4][0],orion[4][1]),(orion[3][0],orion[3][1]),(255,255,255), thickness=1)
      pred_answer = "オリオン座"


    if cyg == 1:
      cv2.line(img,(cygnus[8][0],cygnus[8][1]),(cygnus[7][0],cygnus[7][1]),(255,255,255), thickness=1)
      cv2.line(img,(cygnus[7][0],cygnus[7][1]),(cygnus[0][0],cygnus[0][1]),(255,255,255), thickness=1)
      cv2.line(img,(cygnus[0][0],cygnus[0][1]),(cygnus[1][0],cygnus[1][1]),(255,255,255), thickness=1)
      cv2.line(img,(cygnus[1][0],cygnus[1][1]),(cygnus[2][0],cygnus[2][1]),(255,255,255), thickness=1)#lwing
      cv2.line(img,(cygnus[7][0],cygnus[7][1]),(cygnus[3][0],cygnus[3][1]),(255,255,255), thickness=1)
      cv2.line(img,(cygnus[3][0],cygnus[3][1]),(cygnus[4][0],cygnus[4][1]),(255,255,255), thickness=1)#rwing
      cv2.line(img,(cygnus[7][0],cygnus[7][1]),(cygnus[5][0],cygnus[5][1]),(255,255,255), thickness=1)
      cv2.line(img,(cygnus[5][0],cygnus[5][1]),(cygnus[6][0],cygnus[6][1]),(255,255,255), thickness=1)#head
      pred_answer = "白鳥座"

    if tau ==1:
      cv2.line(img,(taurus[0][0],taurus[0][1]),(taurus[8][0],taurus[8][1]),(255,255,255), thickness=1)
      cv2.line(img,(taurus[1][0],taurus[1][1]),(taurus[8][0],taurus[8][1]),(255,255,255), thickness=1)
      cv2.line(img,(taurus[1][0],taurus[1][1]),(taurus[2][0],taurus[2][1]),(255,255,255), thickness=1)
      cv2.line(img,(taurus[3][0],taurus[3][1]),(taurus[2][0],taurus[2][1]),(255,255,255), thickness=1)
      cv2.line(img,(taurus[3][0],taurus[3][1]),(taurus[4][0],taurus[4][1]),(255,255,255), thickness=1)
      cv2.line(img,(taurus[5][0],taurus[5][1]),(taurus[4][0],taurus[4][1]),(255,255,255), thickness=1)
      cv2.line(img,(taurus[5][0],taurus[5][1]),(taurus[6][0],taurus[6][1]),(255,255,255), thickness=1)
      cv2.line(img,(taurus[7][0],taurus[7][1]),(taurus[6][0],taurus[6][1]),(255,255,255), thickness=1)
      cv2.line(img,(taurus[7][0],taurus[7][1]),(taurus[2][0],taurus[2][1]),(255,255,255), thickness=1)
      cv2.line(img,(taurus[6][0],taurus[6][1]),(taurus[9][0],taurus[9][1]),(255,255,255), thickness=1)
      pred_answer = "おうし座"

    if can == 1:
      cv2.line(img,(cannis[0][0],cannis[0][1]),(cannis[1][0],cannis[1][1]),(255,255,255), thickness=1)
      cv2.line(img,(cannis[0][0],cannis[0][1]),(cannis[2][0],cannis[2][1]),(255,255,255), thickness=1)
      cv2.line(img,(cannis[2][0],cannis[2][1]),(cannis[1][0],cannis[1][1]),(255,255,255), thickness=1)
      cv2.line(img,(cannis[2][0],cannis[2][1]),(cannis[14][0],cannis[14][1]),(255,255,255), thickness=1)
      cv2.line(img,(cannis[3][0],cannis[3][1]),(cannis[14][0],cannis[14][1]),(255,255,255), thickness=1)
      cv2.line(img,(cannis[3][0],cannis[3][1]),(cannis[4][0],cannis[4][1]),(255,255,255), thickness=1)
      cv2.line(img,(cannis[5][0],cannis[5][1]),(cannis[4][0],cannis[4][1]),(255,255,255), thickness=1)
      cv2.line(img,(cannis[5][0],cannis[5][1]),(cannis[6][0],cannis[6][1]),(255,255,255), thickness=1)
      cv2.line(img,(cannis[4][0],cannis[4][1]),(cannis[7][0],cannis[7][1]),(255,255,255), thickness=1)
      cv2.line(img,(cannis[15][0],cannis[15][1]),(cannis[7][0],cannis[7][1]),(255,255,255), thickness=1)
      cv2.line(img,(cannis[15][0],cannis[15][1]),(cannis[9][0],cannis[9][1]),(255,255,255), thickness=1)
      cv2.line(img,(cannis[15][0],cannis[15][1]),(cannis[10][0],cannis[10][1]),(255,255,255), thickness=1)
      cv2.line(img,(cannis[15][0],cannis[15][1]),(cannis[10][0],cannis[10][1]),(255,255,255), thickness=1)
      cv2.line(img,(cannis[7][0],cannis[7][1]),(cannis[8][0],cannis[8][1]),(255,255,255), thickness=1)
      cv2.line(img,(cannis[11][0],cannis[11][1]),(cannis[8][0],cannis[8][1]),(255,255,255), thickness=1)
      cv2.line(img,(cannis[11][0],cannis[11][1]),(cannis[12][0],cannis[12][1]),(255,255,255), thickness=1)
      cv2.line(img,(cannis[13][0],cannis[13][1]),(cannis[11][0],cannis[11][1]),(255,255,255), thickness=1)
      cv2.line(img,(cannis[14][0],cannis[14][1]),(cannis[11][0],cannis[11][1]),(255,255,255), thickness=1)
      pred_answer = "おおいぬ座"

    if ari==1:
      cv2.line(img,(aries[0][0],aries[0][1]),(aries[2][0],aries[2][1]),(255,255,255), thickness=1)
      cv2.line(img,(aries[2][0],aries[2][1]),(aries[3][0],aries[3][1]),(255,255,255), thickness=1)
      cv2.line(img,(aries[3][0],aries[3][1]),(aries[1][0],aries[1][1]),(255,255,255), thickness=1)
      pred_answer = "おひつじ座"
    if gem==1:
      cv2.line(img,(gemeni[15][0],gemeni[15][1]),(gemeni[0][0],gemeni[0][1]),(255,255,255), thickness=1)
      cv2.line(img,(gemeni[1][0],gemeni[1][1]),(gemeni[0][0],gemeni[0][1]),(255,255,255), thickness=1)
      cv2.line(img,(gemeni[2][0],gemeni[2][1]),(gemeni[0][0],gemeni[0][1]),(255,255,255), thickness=1)
      cv2.line(img,(gemeni[2][0],gemeni[2][1]),(gemeni[3][0],gemeni[3][1]),(255,255,255), thickness=1)
      cv2.line(img,(gemeni[5][0],gemeni[5][1]),(gemeni[2][0],gemeni[2][1]),(255,255,255), thickness=1)
      cv2.line(img,(gemeni[5][0],gemeni[5][1]),(gemeni[6][0],gemeni[6][1]),(255,255,255), thickness=1)
      cv2.line(img,(gemeni[4][0],gemeni[4][1]),(gemeni[3][0],gemeni[3][1]),(255,255,255), thickness=1)
      cv2.line(img,(gemeni[0][0],gemeni[0][1]),(gemeni[7][0],gemeni[7][1]),(255,255,255), thickness=1)
      cv2.line(img,(gemeni[8][0],gemeni[8][1]),(gemeni[7][0],gemeni[7][1]),(255,255,255), thickness=1)
      cv2.line(img,(gemeni[8][0],gemeni[8][1]),(gemeni[16][0],gemeni[16][1]),(255,255,255), thickness=1)
      cv2.line(img,(gemeni[8][0],gemeni[8][1]),(gemeni[9][0],gemeni[9][1]),(255,255,255), thickness=1)
      cv2.line(img,(gemeni[8][0],gemeni[8][1]),(gemeni[10][0],gemeni[10][1]),(255,255,255), thickness=1)
      cv2.line(img,(gemeni[11][0],gemeni[11][1]),(gemeni[10][0],gemeni[10][1]),(255,255,255), thickness=1)
      cv2.line(img,(gemeni[12][0],gemeni[12][1]),(gemeni[10][0],gemeni[10][1]),(255,255,255), thickness=1)
      cv2.line(img,(gemeni[12][0],gemeni[12][1]),(gemeni[13][0],gemeni[13][1]),(255,255,255), thickness=1)
      cv2.line(img,(gemeni[14][0],gemeni[14][1]),(gemeni[13][0],gemeni[13][1]),(255,255,255), thickness=1)
      pred_answer = "ふたご座"
    """
    if gem == 1:
      test_num = 0
      for test_p in gemeni:
        cv2.putText(img, str(test_num), (test_p[0], test_p[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), thickness=2)
        test_num = test_num + 1
    """
    dt_now = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
    img_path = img_dir + dt_now + ".jpg"
    cv2.imwrite(img_path, img)
    if(ori==1 or cyg==1 or tau==1 or can==1 or ari==1 or gem==1):
      answer = pred_answer
      return render_template('index.html',img_path=img_path,answer = answer,img_origin =img_origin)
  

  return render_template('index.html',img_path=img_path,answer = "")

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    app.run(host ='0.0.0.0',port = port)