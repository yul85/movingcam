import bpy
import os
import random
import numpy as np
import math
import argparse
import sys


character_info_dict = {
    'adam': ('mixamorig7:', 1.),
    'bryce': ('mixamorig:', 1.),
    'douglas': ('mixamorig:', 1.),
    'jody': ('mixamorig6:', 1.),
    'josh': ('mixamorig:', 1.),
    'leonard': ('mixamorig9:', 1.),
    'lewis': ('mixamorig4:', 1.),
    'liam': ('mixamorig:', 0.5),
    'lola': ('', 1.),
    'malcolm': ('mixamorig:', 0.5),
    'martha': ('mixamorig11:', 1.),
    'megan': ('mixamorig2:', 1.),
    'regina': ('mixamorig:', 0.5),
    'remy': ('mixamorig:', 0.5),
    'shae': ('mixamorig:', 0.5),
    'sophie': ('mixamorig:', 1.),
    'stefani': ('mixamorig:', 0.5)
}


def rotX(theta):
    R = np.eye(3)
    c = math.cos(theta)
    s = math.sin(theta)
    R[1,1]=c; R[1,2]=-s
    R[2,1]=s; R[2,2]=c
    return R


def rotY(theta):
    R = np.eye(3)
    c = math.cos(theta)
    s = math.sin(theta)
    R[0,0]=c; R[0,2]=s
    R[2,0]=-s; R[2,2]=c
    return R


def rotZ(theta):
    R = np.eye(3)
    c = math.cos(theta)
    s = math.sin(theta)
    R[0,0]=c; R[0,1]=-s
    R[1,0]=s; R[1,1]=c
    return R


def set_cycles_renderer(scene: bpy.types.Scene,
                        camera_object: bpy.types.Object,
                        num_samples: int,
                        use_denoising: bool = True,
                        use_motion_blur: bool = False,
                        use_transparent_bg: bool = False) -> None:
    scene.camera = camera_object

    scene.render.image_settings.file_format = 'PNG'
    scene.render.engine = 'CYCLES'
    scene.render.use_motion_blur = use_motion_blur

    scene.render.film_transparent = use_transparent_bg
    scene.view_layers[0].cycles.use_denoising = use_denoising

    scene.cycles.samples = num_samples


def copy_transform(armature_name, bone_name, rotation=True):
    cmd = 'COPY_ROTATION' if rotation else 'COPY_LOCATION'
    context = bpy.context
    scene = bpy.context.scene

    bone = scene.objects['Armature'].pose.bones[bone_name]
    bone2 = None
    # other armature other bone
    arm2 = scene.objects.get(armature_name)
    if arm2 is not None:
        bone2 = arm2.pose.bones.get(bone_name)
    else:
        print('arm2 is None')
    # give it a copy rotation constraint
    if bone is not None:
        crc = bone.constraints.new(cmd)
        # give it a target bone
        crc.target = arm2
        # note subtarget uses name not object.
        crc.subtarget = bone2.name
    else:
        print('bone is None')


def remove_constraints(context):
    for i in range(len(context.object.pose.bones)):
        bone = context.object.pose.bones[i]
        # Create a list of all the copy location constraints on this bone
        copyLocConstraints = [c for c in bone.constraints if c.type == 'COPY_ROTATION' or c.type == 'COPY_LOCATION']

        # Iterate over all the bone's copy location constraints and delete them all
        for c in copyLocConstraints:
            bone.constraints.remove(c)  # Remove constraint


def main(context):
    parser = argparse.ArgumentParser()
    parser.add_argument('--character_name', type=str)
    parser.add_argument('--motion_name', type=str)
    parser.add_argument('--render_path', type=str)
    parser.add_argument('--camera_move_type', type=str, choices=['Fix', 'R_track', 'T_track'], default='Fix')
    parser.add_argument('--camera_track_param', type=int, nargs=2, default=[0, 0])
    args = parser.parse_args(sys.argv[6:])

    scene = context.scene

    # import os, random
    mat = bpy.data.materials['ground']
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    ps = nodes.get('Principled BSDF')
    mat_out = nodes.get('Material Output')

    texture_dir = "/home/user/works/mixamo_render/ground_textures"
    texture_list = os.listdir(texture_dir)
    texture_idx = random.randrange(len(texture_list))
    texture_name = texture_list[texture_idx]
    texture_file_list = os.listdir(texture_dir + "/" + texture_name)
    color_file_name = ''
    normal_file_name = ''
    roughness_file_name = ''
    specular_file_name = ''
    disp_file_name = ''
    metalic_file_name = ''

    for filename in texture_file_list:
        if 'COL' in filename.upper():
            color_file_name = filename
        if 'NRM' in filename.upper() or 'NORM' in filename.upper():
            normal_file_name = filename
        if 'SPEC' in filename.upper():
            specular_file_name = filename
        if 'ROUGH' in filename.upper():
            roughness_file_name = filename
        if 'DISP' in filename.upper() or 'HEIGHT' in filename.upper():
            disp_file_name = filename
        if 'METALIC' in filename.upper():
            metalic_file_name = filename

    # for scaling
    texcoord_node = nodes.new('ShaderNodeTexCoord')
    mapping_node = nodes.new('ShaderNodeMapping')
    mapping_node.vector_type = 'TEXTURE'
    links.new(texcoord_node.outputs[2], mapping_node.inputs[0])
    mapping_node.inputs['Scale'].default_value = (0.01, 0.01, 1.)

    # color
    color_node = nodes.new('ShaderNodeTexImage')
    links.new(mapping_node.outputs[0], color_node.inputs[0])
    bpy.ops.image.open(filepath=texture_dir + '/' + texture_name + '/' + color_file_name)
    bpy.data.images[color_file_name].pack()
    color_node.image = bpy.data.images[color_file_name]
    # color_node.image.colorspace_settings.name = 'Non-Color'
    color_node.image.colorspace_settings.name = 'sRGB'
    links.new(color_node.outputs[0], ps.inputs[0])

    # specular
    if specular_file_name:
        specular_node = nodes.new('ShaderNodeTexImage')
        bpy.ops.image.open(filepath=texture_dir + '/' + texture_name + '/' + specular_file_name)
        bpy.data.images[specular_file_name].pack()
        specular_node.image = bpy.data.images[specular_file_name]
        specular_node.image.colorspace_settings.name = 'Non-Color'
        links.new(mapping_node.outputs[0], specular_node.inputs[0])
        links.new(specular_node.outputs[0], ps.inputs[7])

    # roughness
    if roughness_file_name:
        roughness_node = nodes.new('ShaderNodeTexImage')
        bpy.ops.image.open(filepath=texture_dir + '/' + texture_name + '/' + roughness_file_name)
        bpy.data.images[roughness_file_name].pack()
        roughness_node.image = bpy.data.images[roughness_file_name]
        roughness_node.image.colorspace_settings.name = 'Non-Color'
        links.new(mapping_node.outputs[0], roughness_node.inputs[0])
        links.new(roughness_node.outputs[0], ps.inputs[7])

    # normal
    normal_node = nodes.new('ShaderNodeTexImage')
    bpy.ops.image.open(filepath=texture_dir + '/' + texture_name + '/' + normal_file_name)
    bpy.data.images[normal_file_name].pack()
    normal_node.image = bpy.data.images[normal_file_name]
    normal_node.image.colorspace_settings.name = 'Non-Color'
    links.new(mapping_node.outputs[0], normal_node.inputs[0])
    normal_map_node = nodes.new('ShaderNodeNormalMap')
    normal_map_node.uv_map = 'UVMap'
    links.new(normal_node.outputs[0], normal_map_node.inputs[1])
    links.new(normal_map_node.outputs[0], ps.inputs[19])

    # metailc
    if metalic_file_name:
        metalic_node = nodes.new('ShaderNodeTexImage')
        bpy.ops.image.open(filepath=texture_dir + '/' + texture_name + '/' + metalic_file_name)
        bpy.data.images[metalic_file_name].pack()
        metalic_node.image = bpy.data.images[metalic_file_name]
        metalic_node.image.colorspace_settings.name = 'Non-Color'
        links.new(mapping_node.outputs[0], metalic_node.inputs[0])
        links.new(metalic_node.outputs[0], ps.inputs[4])

    # displacement
    if disp_file_name:
        disp_node = nodes.new('ShaderNodeTexImage')
        bpy.ops.image.open(filepath=texture_dir + '/' + texture_name + '/' + disp_file_name)
        bpy.data.images[disp_file_name].pack()
        disp_node.image = bpy.data.images[disp_file_name]
        disp_node.image.colorspace_settings.name = 'Non-Color'
        links.new(mapping_node.outputs[0], disp_node.inputs[0])
        disp_map_node = nodes.new('ShaderNodeDisplacement')
        links.new(disp_node.outputs[0], disp_map_node.inputs[0])
        links.new(disp_map_node.outputs[0], mat_out.inputs[2])

    # Light setting
    light_off_num = np.random.randint(0, 4)
    light_off_seq = np.random.permutation(4)
    if light_off_num > 0:
        for light_idx in range(light_off_num):
            light_object = bpy.context.scene.objects['Light.00' + str(light_off_seq[light_idx])]
            light_data = bpy.data.lights.get('Light.00'+str(light_off_seq[light_idx]))
            light_object.hide_render = True
            light_data.energy = 500. + 1000. * np.clip(np.random.randn(), 0., 1.)

    # euler_angles = [camera.rotation_euler[i] for i in range(3)]
    # rotation_matrix = np.dot(np.dot(rotX(euler_angles[0]), rotY(euler_angles[1])), rotZ(euler_angles[2]))

    character_name = args.character_name
    character_info = character_info_dict[character_name]
    bpy.ops.import_scene.fbx(filepath="/home/user/works/mixamo_render/characters/" + character_name + ".fbx", global_scale=character_info[1])

    motion_name = args.motion_name
    bpy.ops.import_scene.fbx(filepath="/data/mixamo_motions/mixamo_" + character_name + "/" + motion_name + ".fbx", global_scale=character_info[1])
    object = scene.objects['Armature']
    
    character_bone_prefix = ''
    for i in range(len(object.pose.bones)):
        name = object.pose.bones[i].name
        if 'Hips' in name:
            character_bone_prefix = name[:name.find('Hips')]
            break

    copy_transform('Armature.001', character_bone_prefix + 'Hips', False)
    for i in range(len(object.pose.bones)):
        name = object.pose.bones[i].name
        copy_transform('Armature.001', name)

    keys = None
    if bpy.data.actions:
        # get all actions
        action_list = [action.frame_range for action in bpy.data.actions]

        # sort, remove doubles and create a set
        keys = (sorted(set([item for sublist in action_list for item in sublist])))

        # print all keyframes
        print(keys)

        # print first and last keyframe
        print("{} {}".format("first keyframe:", keys[0]))
        print("{} {}".format("last keyframe:", keys[-1]))

        # assign new starting frame
        scene.frame_start = keys[0]

        # assign new end frame
        scene.frame_end = keys[-1]

    # make cube
    bpy.ops.mesh.primitive_cube_add()
    bpy.context.scene.objects['Cube'].data.name = 'hips'
    bpy.context.scene.objects['Cube'].name = 'hips'
    # constraint
    crc = bpy.context.scene.objects['hips'].constraints.new('COPY_LOCATION')
    crc.target = bpy.context.scene.objects['Armature']
    crc.subtarget = character_bone_prefix+'Hips'
    bpy.context.scene.objects["Armature"].select_set(True)
    bpy.context.scene.objects['hips'].select_set(True)
    bpy.ops.nla.bake(frame_start=scene.frame_start, frame_end=scene.frame_end, visual_keying=True, clear_constraints=True, only_selected=False, bake_types={'OBJECT'})

    # Camera setting
    hip_xyz_range = [[100000., -100000.], [100000., -100000.], [100000., -100000.]]
    hip_location_data = []
    for i in range(scene.frame_start, scene.frame_end+1):
        bpy.context.scene.frame_set(i)
        hip_location = bpy.context.scene.objects['hips'].location
        hip_location_data.append(hip_location)
        for j in range(3):
            hip_xyz_range[j][0] = min(hip_xyz_range[j][0], hip_location[j])
            hip_xyz_range[j][1] = max(hip_xyz_range[j][1], hip_location[j])
    bpy.context.scene.objects['hips'].hide_render = True

    bpy.context.scene.frame_set(1)

    bpy.data.cameras[0].lens = 50.
    camera = bpy.context.scene.objects['Camera']

    camera.location[0] = 0.
    camera.location[1] = 0.
    camera.location[2] = 1.
    # rotation is euler XYZ with rad
    camera.rotation_euler[0] = 0.
    camera.rotation_euler[1] = 0.
    camera.rotation_euler[2] = 0.

    camera_distance = 7.5 + 3. * np.random.rand()
    if args.camera_move_type == 'Fix':
        height = np.clip(1.0 + 0.6 * np.random.randn(), 0.3, 2.0)
        camera_xy_distance = math.sqrt(camera_distance ** 2 - (height-1.) ** 2)
        camera_z_angle = np.random.rand() * 2. * math.pi
        camera_x_angle = math.acos((height-1.)/camera_distance)
        camera_xy_vector = np.dot(rotZ(camera_z_angle), np.array([0., -camera_xy_distance, 0.]))
        camera.location[0] = camera_xy_vector[0] + sum(hip_xyz_range[0])/2.
        camera.location[1] = camera_xy_vector[1] + sum(hip_xyz_range[1])/2.
        camera.location[2] = height
        camera.rotation_euler[0] = camera_x_angle
        camera.rotation_euler[2] = camera_z_angle
    elif args.camera_move_type == 'R_track':
        height = np.clip(1.0 + 0.6 * np.random.randn(), 0.3, 2.0)
        camera_xy_distance = math.sqrt(camera_distance ** 2 - (height-1.) ** 2)
        camera_z_angle = np.random.randint(0, 2) * math.pi
        camera_x_range = hip_xyz_range[0][1] - hip_xyz_range[0][0]
        camera_y_range = hip_xyz_range[1][1] - hip_xyz_range[1][0]
        if camera_x_range < camera_y_range:
            camera_z_angle += math.pi/2.
        camera_x_angle = math.acos((height-1.)/camera_distance)
        camera_xy_vector = np.dot(rotZ(camera_z_angle), np.array([0., -camera_xy_distance, 0.]))
        camera.location[0] = camera_xy_vector[0]
        camera.location[1] = camera_xy_vector[1]
        camera.location[2] = height
        camera.rotation_euler[0] = camera_x_angle
        camera.rotation_euler[2] = camera_z_angle
        crc = camera.constraints.new('DAMPED_TRACK')
        crc.target = bpy.context.scene.objects["Armature"]
        crc.subtarget = character_bone_prefix+'Hips'
        crc.track_axis = 'TRACK_NEGATIVE_Z'
    elif args.camera_move_type == 'T_track':
        camera_x_angle, camera_z_angle = args.camera_track_param
        camera_x_angle = 90 - camera_x_angle
        camera_x_angle = camera_x_angle/180 * math.pi
        camera_z_angle = camera_z_angle/180 * math.pi
        height = camera_distance * math.cos(camera_x_angle) + 0.5 * (hip_xyz_range[2][0] + hip_xyz_range[2][1])
        camera_xy_distance = camera_distance * math.sin(camera_x_angle)
        camera_xy_vector = np.dot(rotZ(camera_z_angle), np.array([0., -camera_xy_distance, 0.]))
        camera.location[0] = camera_xy_vector[0]
        camera.location[1] = camera_xy_vector[1]
        camera.location[2] = height
        camera.rotation_euler[0] = camera_x_angle
        camera.rotation_euler[2] = camera_z_angle
        crc = camera.constraints.new('COPY_LOCATION')
        crc.target = bpy.context.scene.objects['Armature']
        crc.subtarget = character_bone_prefix+'Hips'
        crc.use_offset = True
        crc.use_z = False

    render_path = args.render_path + '/'

    # Set render resolution
    scene.render.engine = 'BLENDER_EEVEE'
    # scene.render.engine = 'CYCLES'
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 720
    scene.render.resolution_percentage = 100
    scene.eevee.use_motion_blur = True
    # scene.eevee.use_motion_blur = False

    # Render to separate file, identified by texture file
    scene.render.filepath = render_path

    # Render still image, automatically write to output path
    bpy.ops.render.render(animation=True)


class SimpleOperator(bpy.types.Operator):
    """Tooltip"""
    bl_idname = "object.simple_operator"
    bl_label = "Simple Object Operator"

    @classmethod
    def poll(cls, context):
        return context.active_object is not None

    def execute(self, context):
        main(context)
        return {'FINISHED'}


def register():
    bpy.utils.register_class(SimpleOperator)


def unregister():
    bpy.utils.unregister_class(SimpleOperator)


if __name__ == "__main__":
    register()

    # test call
    bpy.ops.object.simple_operator()
