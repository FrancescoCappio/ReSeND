import os

office31 = {'backpack':0, "bike":1,"bike_helmet":2,"bookcase":3,"bottle":4,"calculator":5,
            "desk_chair":6,"desk_lamp":7,"desktop_computer":8,"file_cabinet":9,"headphones":10,
            "keyboard":11,"laptop":12,"letter_tray":13,"mobile_phone":14,"monitor":15,
            "mouse":16,"mug":17,"paper_notebook":18,"pen":19,"phone":20,
            "printer":21,"projector":22,"punchers":23,"ring_binder":24,"ruler":25,
            "scissors":26,"speaker":27,"stapler":28,"tape_dispenser":29,"trash_can":30}


VisDA ={'bicycle':1,'aeroplane':31,'bus':32,'car':33,'horse':34,'knife':35,'motorcycle':36,'person':37,
        'plant':38,'skateboard':39,'train':40,'truck':41}

STL10 = {'airplane':31,'car':33,'horse':34,'truck':41,'bird':42,'cat':43,'deer':44,
         'dog':45,'monkey':46,'ship':47}


domainnet = {'backpack':0,'bicycle':1,'calculator':5,'chair':6,'headphones':10,
             'keyboard':11,'cell_phone':14,'mug':17,'telephone':20,'scissors':26,
             'airplane':31,'bus':32,'car':33,'horse':34,'knife':35,'motorbike':36,
             'skateboard':39,'train':40,'truck':41,'bird':42,'cat':43,
             'dog':45,'monkey':46,'ant':48,'apple':49,'banana':50,
             'basket':51,'butterfly':52,'candle':53,'cookie':54,'eyeglasses':55,
             'grapes':56,'ice_cream':57,'mailbox':58,'necklace':59,'octopus':60,
             'piano':61,'potato':62,'roller_coaster':63,'snail':64,'snowman':65,
             'star':66,'violin':67}



target = 'Cartoon'




if target=='ArtPainting':
    sources = ['Cartoon','Photo','Sketch']
if target=='Sketch':
    sources = ['Photo','ArtPainting','Cartoon']
if target=='Photo':
    sources = ['ArtPainting','Cartoon','Sketch']
if target=='Cartoon':
    sources = ['Sketch','Photo','ArtPainting']

classes_S1 = [3,0,1]
classes_S2 = [4,0,2]
classes_S3 = [5,1,2]

print('Target: ',target)
print('Sources: ',sources[0],sources[1],sources[2])

source_1 = open('/work/sbucci/DomainShift_CategoryShift/data/txt_lists/PACS_DG/'+sources[0]+'.txt','r').readlines()
source_2 = open('/work/sbucci/DomainShift_CategoryShift/data/txt_lists/PACS_DG/'+sources[1]+'.txt','r').readlines()
source_3 = open('/work/sbucci/DomainShift_CategoryShift/data/txt_lists/PACS_DG/'+sources[2]+'.txt','r').readlines()

new_file = open('/work/sbucci/DomainShift_CategoryShift/data/txt_lists/PACS_DG/no_'+target+'.txt','w')

for line in source_1:
    label_class = int(line.split(' ')[1].strip())
    if label_class in classes_S1:
        new_file.write(line.strip()+' '+str(0)+'\n')

for line in source_2:
    label_class = int(line.split(' ')[1].strip())
    if label_class in classes_S2:
        new_file.write(line.strip()+' '+str(1)+'\n')

for line in source_3:
    label_class = int(line.split(' ')[1].strip())
    if label_class in classes_S3:
        new_file.write(line.strip()+' '+str(2)+'\n')

