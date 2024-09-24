import shutil, os, random, torch, sys
import numpy as np
import mitsuba as mi
mi.set_variant("scalar_rgb")
from mitsuba import ScalarTransform4f as T
from matplotlib import pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../'))

from datasets.io import IO
from utils import misc

import uuid

xml_head = \
"""
<scene version="0.6.0">
    <integrator type="path">
        <integer name="maxDepth" value="-1"/>
    </integrator>
    <sensor type="perspective">
        <float name="farClip" value="100"/>
        <float name="nearClip" value="0.1"/>
        <transform name="toWorld">
            <lookat origin="3,3,3" target="0,0,0" up="0,0,1"/>
        </transform>
        <float name="fov" value="25"/>
        
        <sampler type="ldsampler">
            <integer name="sampleCount" value="256"/>
        </sampler>
        <film type="hdrfilm">
            <integer name="width" value="1600"/>
            <integer name="height" value="1200"/>
            <rfilter type="gaussian"/>
            <boolean name="banner" value="false"/>
        </film>
    </sensor>
    
    <bsdf type="roughplastic" id="surfaceMaterial">
        <string name="distribution" value="ggx"/>
        <float name="alpha" value="0.05"/>
        <float name="intIOR" value="1.46"/>
        <rgb name="diffuseReflectance" value="1,1,1"/> <!-- default 0.5 -->
    </bsdf>
    
"""

xml_ball_segment = \
"""
    <shape type="sphere">
        <float name="radius" value="0.01"/>
        <transform name="toWorld">
            <translate x="{}" y="{}" z="{}"/>
        </transform>
        <bsdf type="diffuse">
            <rgb name="reflectance" value="{},{},{}"/>
        </bsdf>
    </shape>
"""

xml_tail = \
"""
    <shape type="rectangle">
        <ref name="bsdf" id="surfaceMaterial"/>
        <transform name="toWorld">
            <scale x="10" y="10" z="1"/>
            <translate x="0" y="0" z="-0.5"/>
        </transform>
    </shape>
    
    <shape type="rectangle">
        <transform name="toWorld">
            <scale x="10" y="10" z="1"/>
            <lookat origin="-4,4,20" target="0,0,0" up="0,0,1"/>
        </transform>
        <emitter type="constant">
            <rgb name="radiance" value="6.0"/>
        </emitter>
    </shape>
</scene>
"""

def pc55_norm(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def standardize_bbox(pcl, points_per_object):
    pt_indices = np.random.choice(pcl.shape[0], points_per_object, replace=False)
    np.random.shuffle(pt_indices)
    pcl = pcl[pt_indices] # n by 3
    mins = np.amin(pcl, axis=0)
    maxs = np.amax(pcl, axis=0)
    center = ( mins + maxs ) / 2.
    scale = np.amax(maxs-mins)
    # print("Center: {}, Scale: {}".format(center, scale))
    result = ((pcl - center)/scale).astype(np.float32) # [-0.5, 0.5]
    return result


def colormap(x,y,z):
    vec = np.array([x,y,z])
    vec = np.clip(vec, 0.001,1.0)
    norm = np.sqrt(np.sum(vec**2))
    vec /= norm
    return [vec[0], vec[1], vec[2]]


def load_sensor(r, phi, theta):
    origin = T.rotate([0, 0, 1], phi).rotate([0, 1, 0], theta) @ mi.ScalarPoint3f([0, 0, r])
    return mi.load_dict({
        'type': 'perspective',
        'fov': 39.3077,
        'to_world': T.look_at(
            origin=origin,
            target=[0, 0, 0],
            up=[0, 0, 1]
        ),
        'sampler': {
            'type': 'independent',
            'sample_count': 16
        },
        'film': {
            'type': 'hdrfilm',
            'width': 1600,
            'height': 1200,
            'rfilter': {
                'type': 'tent',
            },
            'pixel_format': 'rgb',
        },
    })
    

def copy55():
        # Open file    
    fileHandler  =  open  ("data/ShapeNet55-34/ShapeNet-55/test.txt",  "r")
    # Get list of all lines in file
    listOfLines  =  fileHandler.readlines()
    for  line in  listOfLines:
        filename1 = os.path.join("data/ShapeNet55-34/shapenet_pc", line.strip())
        shutil.copy(filename1, "demo/demo55")


    # Close file
    fileHandler.close()
    # Iterate over the lines

def sample55():
    files = [os.path.join('demo/demo_55/gt_all', f) for f in os.listdir('demo/demo_55/gt_all')]
    random_files = random.sample(files, 200)
    npoints = 8192
    for file_path in random_files:

        pc_ndarray = IO.get(file_path).astype(np.float32)
        pc_ndarray = pc55_norm(pc_ndarray)
        pc_ndarray = torch.tensor(pc_ndarray).unsqueeze(0).cuda()

        partial, _ = misc.seprate_point_cloud(pc_ndarray, npoints, [int(npoints * 1/4) , int(npoints * 3/4)], fixed_points = None)
        partial1 = np.array(partial.squeeze().cpu())
        pc_ndarray1 = np.array(pc_ndarray.squeeze().cpu())
        if not os.path.exists('demo/demo_55/partial'):
            os.makedirs('demo/demo_55/partial')
        if not os.path.exists('demo//demo_55/gt'):
            os.makedirs('demo/demo_55/gt')
        p = os.path.join('demo/demo_55/partial/', os.path.basename(file_path).split('.')[0]+'.txt') 
        g = os.path.join('demo/demo_55/gt/', os.path.basename(file_path).split('.')[0]+'.txt') 
        np.savetxt(p, partial1,fmt='%.32f')
        np.savetxt(g, pc_ndarray1,fmt='%.32f')


def copypcn():
    file1 = [os.path.join('data/PCN/val/partial/', f) for f in os.listdir('data/PCN/val/partial/')]
    for f1 in file1:
        file2 = [os.path.join(f1, f) for f in os.listdir(f1)]
        for f2 in file2:
            filep = os.path.join(f2, '00.pcd')
            shutil.copy(filep, 'demo/demo_pcn/partial_all/'+ os.path.basename(f1)+'_'+os.path.basename(f2)+'.pcd')
            fileg = f2.replace("partial","complete")+".pcd"
            shutil.copy(fileg, 'demo/demo_pcn/gt_all/'+ os.path.basename(f1)+'_'+os.path.basename(f2)+'.pcd')


def samplepcn():
    files = [os.path.join('demo/demo_pcn/partial_all', f) for f in os.listdir('demo/demo_pcn/partial_all')]
    random_files = random.sample(files, 100)
    for file_path in random_files:
        fileg = file_path.replace("partial","gt")
        partial = torch.tensor(IO.get(file_path).astype(np.float32)).unsqueeze(0).cuda()
        gt = torch.tensor(IO.get(fileg).astype(np.float32)).unsqueeze(0).cuda()
        partial = np.array(partial.squeeze().cpu())
        gt = np.array(gt.squeeze().cpu())
        if not os.path.exists('demo/demo_pcn/partial'):
            os.makedirs('demo/demo_pcn/partial')
        if not os.path.exists('demo/demo_pcn/gt'):
            os.makedirs('demo/demo_pcn/gt')
        p = os.path.join('demo/demo_pcn/partial/', os.path.basename(file_path).split('.')[0]+'.txt') 
        g = os.path.join('demo/demo_pcn/gt/', os.path.basename(file_path).split('.')[0]+'.txt') 
        np.savetxt(p, partial,fmt='%.32f')
        np.savetxt(g, gt,fmt='%.32f')

# Kiiti数据集要处理的多，自己不好处理，推断的时候留一份副本备好，见data_transformer.py 行 138
def CopyKitti(pc):
    random_filename = str(uuid.uuid4())
    f1 = 'demo/demo_kitti/partial_all/' + random_filename + '.txt'
    print(f1)
    np.savetxt(f1, pc ,fmt='%.32f')

def render():
    file1 = [os.path.join('demo/demo_55/result', f) for f in os.listdir('demo/demo_55/result')]
    count = 0
    for f1 in file1:
        file2 = [os.path.join(f1, f) for f in os.listdir(f1)]
        for f2 in file2:
            if os.path.basename(f2).split('.')[-1] == 'txt':
                    xml_segments = [xml_head]
                    pcl = IO.get(f2).astype(np.float32)
                    pcl = standardize_bbox(pcl, 8192)
                    pcl = pcl[:,[2,0,1]]
                    pcl[:,0] *= -1
                    pcl[:,2] += 0.0125
                    for i in range(pcl.shape[0]):
                        color = colormap(pcl[i,0]+0.5,pcl[i,1]+0.5,pcl[i,2]+0.5-0.0125)
                        xml_segments.append(xml_ball_segment.format(pcl[i,0],pcl[i,1],pcl[i,2], *color))
                    xml_segments.append(xml_tail)
                    xml_content = str.join('', xml_segments)
                    with open(f2 + '.xml', 'w') as f:
                        f.write(xml_content)
        print(count)
        count = count+1
        
    '''
    file1 = [os.path.join('demo/demo_55/gt', f) for f in os.listdir('demo/demo_55/gt')]
    for f2 in file1:
        xml_segments = [xml_head]
        pcl = IO.get(f2).astype(np.float32)
        pcl = standardize_bbox(pcl, 8192)
        pcl = pcl[:,[2,0,1]]
        pcl[:,0] *= -1
        pcl[:,2] += 0.0125
        for i in range(pcl.shape[0]):
            color = colormap(pcl[i,0]+0.5,pcl[i,1]+0.5,pcl[i,2]+0.5-0.0125)
            xml_segments.append(xml_ball_segment.format(pcl[i,0],pcl[i,1],pcl[i,2], *color))
        xml_segments.append(xml_tail)
        xml_content = str.join('', xml_segments)
        with open(f2 + '.xml', 'w') as f:
            f.write(xml_content)
    
    file1 = [os.path.join('demo/demo_55/partial', f) for f in os.listdir('demo/demo_55/partial')]
    for f2 in file1:
        xml_segments = [xml_head]
        pcl = IO.get(f2).astype(np.float32)
        pcl = standardize_bbox(pcl, 2048)
        pcl = pcl[:,[2,0,1]]
        pcl[:,0] *= -1
        pcl[:,2] += 0.0125
        for i in range(pcl.shape[0]):
            color = colormap(pcl[i,0]+0.5,pcl[i,1]+0.5,pcl[i,2]+0.5-0.0125)
            xml_segments.append(xml_ball_segment.format(pcl[i,0],pcl[i,1],pcl[i,2], *color))
        xml_segments.append(xml_tail)
        xml_content = str.join('', xml_segments)
        with open(f2 + '.xml', 'w') as f:
            f.write(xml_content)
    '''

def show():
    '''
    file1 = [os.path.join('demo/demo_55/gt', f) for f in os.listdir('demo/demo_55/gt')]
    count = 0
    for f2 in file1:
        if os.path.basename(f2).split('.')[-1] == 'xml':
            sensor_count = 6
            radius = 2
            phis = [20.0 * i for i in range(sensor_count)]
            theta = 60.0
            sensors = [load_sensor(radius, phi, theta) for phi in phis]

            scene = mi.load_file(f2)
            images = [mi.render(scene, sensor=sensor) for sensor in sensors]
            for i in range(sensor_count):
                path = f2.split('.')[0] + '_' + str(i) +'.exr'
                mi.Bitmap(images[i]).write(path)
            count = count+1
            print(count)

    file1 = [os.path.join('demo/demo_55/partial', f) for f in os.listdir('demo/demo_55/partial')]
    count = 0
    for f2 in file1:
        if os.path.basename(f2).split('.')[-1] == 'xml':
            sensor_count = 6
            radius = 2
            phis = [20.0 * i for i in range(sensor_count)]
            theta = 60.0
            sensors = [load_sensor(radius, phi, theta) for phi in phis]

            scene = mi.load_file(f2)
            images = [mi.render(scene, sensor=sensor) for sensor in sensors]
            for i in range(sensor_count):
                path = f2.split('.')[0] + '_' + str(i) +'.exr'
                mi.Bitmap(images[i]).write(path)
            count = count+1
            print(count)
    '''
    file1 = [os.path.join('demo/demo_55/result', f) for f in os.listdir('demo/demo_55/result')]
    count = 0
    for f1 in file1:
        file2 = [os.path.join(f1, f) for f in os.listdir(f1)]
        for f2 in file2:
            if os.path.basename(f2).split('.')[-1] == 'xml':
                    sensor_count = 6
                    radius = 2
                    phis = [20.0 * i for i in range(sensor_count)]
                    theta = 60.0
                    sensors = [load_sensor(radius, phi, theta) for phi in phis]

                    scene = mi.load_file(f2)
                    images = [mi.render(scene, sensor=sensor) for sensor in sensors]
                    for i in range(sensor_count):
                        path = f2.split('.')[0] + '_' + str(i) +'.exr'
                        mi.Bitmap(images[i]).write(path)
        count = count+1
        print(count)

def showone(path, num):
    xml_segments = [xml_head]
    pcl = IO.get(path).astype(np.float32)
    pcl = misc.fps(torch.tensor(pcl).unsqueeze(0).cuda(), num)
    pcl = np.array(pcl.squeeze().cpu())

    pcl = standardize_bbox(pcl, num)
    pcl = pcl[:,[2,0,1]]
    pcl[:,0] *= -1
    pcl[:,2] += 0.0125
    for i in range(pcl.shape[0]):
        color = colormap(pcl[i,0]+0.5,pcl[i,1]+0.5,pcl[i,2]+0.5-0.0125)
        xml_segments.append(xml_ball_segment.format(pcl[i,0],pcl[i,1],pcl[i,2], *color))
    xml_segments.append(xml_tail)
    xml_content = str.join('', xml_segments)
    with open(path + '.xml', 'w') as f:
        f.write(xml_content)

    sensor_count = 6
    radius = 2
    phis = [20.0 * i for i in range(sensor_count)]
    theta = 60.0
    sensors = [load_sensor(radius, phi, theta) for phi in phis]

    scene = mi.load_file(path + '.xml')
    images = [mi.render(scene, sensor=sensor) for sensor in sensors]
    for i in range(sensor_count):
        path1 = path + '_' + str(i) +'.exr'
        mi.Bitmap(images[i]).write(path1)

def fpspoint(path,num):
    pcl = IO.get(path).astype(np.float32)
    pcl = misc.fps(torch.tensor(pcl).unsqueeze(0).cuda(), num)
    pcl = np.array(pcl.squeeze().cpu())
    with open(path +'_'+ str(num) +'.txt', 'w') as f:
        f.write(pcl)





def main():
    # copy55()
    # sample55()
    # copypcn()
    # samplepcn()
    # render()
    # show()
    # showone('demo/fig/2.txt', 640)
    # fpspoint()
    # saveKitti()


    
if __name__ == '__main__':
    main()