import cv2
import numpy as np
import pyrealsense2 as rs

def main():

    pipeline = rs.pipeline()
    config = rs.config()
    colorizer = rs.colorizer()

    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    colorizer.set_option(rs.option.min_distance, 0.4)
    colorizer.set_option(rs.option.max_distance, 0.6)

    pc = rs.pointcloud()
    points = rs.points()

    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)

    profile = pipeline.start(config)

    depth_sensor = profile.get_device().first_depth_sensor()

    # Usa o preset do high accuracy para configurar a câmera
    depth_sensor.set_option(rs.option.visual_preset, 3)

    sensor_dep = profile.get_device().query_sensors()[0]
    sensor_dep.set_option(rs.option.laser_power, 100)

    align_to = rs.stream.color
    align = rs.align(align_to)

    while True:

        frames = pipeline.wait_for_frames()

        ### PARA APRESENTAR IMAGENS
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
            
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        # If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            images = np.hstack((color_image, depth_colormap))

        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        cv2.waitKey(1)
        ###

        # TIRAR POINTCLOUD SE "ESC" É APERTADO
        if input() == "p":
            # Itera o nome do arquivo novo de PC a ser salvo
            f = open("pointclouds/qtdPC.txt", "r+")
            numPC = int(f.readlines()[-1]) + 1
            f.write(str(numPC) + "\n")
            f.close()


            colorized = colorizer.process(frames)

            ply = rs.save_to_ply("pointclouds/pointcloud{}.ply".format(numPC))
            ply.set_option(rs.save_to_ply.option_ply_binary, False)
            ply.set_option(rs.save_to_ply.option_ply_normals, True)
        
            ply.process(colorized)

            print("PROCESSO TERMINADO")

if __name__ == "__main__":
    main()