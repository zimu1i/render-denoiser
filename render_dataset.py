"""
Generate training pairs for a Monte Carlo denoiser, using Blender's Cycles.

Each view produces four images:

    noisy.exr     the fast render the denoiser must clean up   (input)
    albedo.exr    surface colour, free of noise                (input)
    normal.exr    surface orientation, free of noise           (input)
    clean.exr     the converged render                         (target)

Run it headless:

    blender --background scene.blend --python render_dataset.py -- \
        --out data/raw --views 20

or without any scene file, building random ones instead:

    blender --background --python render_dataset.py -- \
        --out data/raw --views 40 --procedural

Four decisions in here are load-bearing; the rest is Blender API plumbing.

1.  Cycles' own denoiser is switched OFF for both renders. If the target were
    denoised by OpenImageDenoise, the network would be trained to imitate OIDN
    rather than the true converged image, and OIDN could never be used as an
    honest baseline to compare against.

2.  The auxiliary buffers come from the NOISY render, not the clean one. At
    inference time only the fast render exists, so taking albedo and normal
    from a converged render would leak information the real system will never
    have -- and would make the reported numbers meaningless.

3.  Both renders in a pair use the same seed and camera, so they are pixel
    aligned. Without that the network is learning to correct a misalignment.

4.  The view transform is set to Raw so the EXR files hold linear radiance.
    Blender's default Filmic/AgX view transform would bake a tone curve into
    what is supposed to be physically linear data.
"""

import argparse
import math
import os
import random
import sys

import bpy
from mathutils import Vector

# Passes that ship as separate EXR files alongside the beauty render. Both are
# essentially noise-free even at low sample counts, which is exactly why
# production denoisers condition on them.
AUX_PASSES = {
    "albedo": "Diffuse Color",
    "normal": "Normal",
}


def ensure_nodes(datablock):
    """Turn on the node tree only if it is not already there.

    Blender 5 creates node trees by default and deprecates `use_nodes` ahead of
    its removal in 6.0, but earlier versions still require it. Checking for the
    tree works on both and keeps the deprecation warning quiet.
    """
    if getattr(datablock, "node_tree", None) is None:
        datablock.use_nodes = True


def parse_args():
    """Read arguments after the '--' that Blender uses to end its own."""
    argv = sys.argv
    argv = argv[argv.index("--") + 1 :] if "--" in argv else []

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/raw", help="output directory")
    parser.add_argument("--views", type=int, default=20, help="camera views to render")
    parser.add_argument("--noisy-spp", type=int, default=4, help="samples for the input")
    parser.add_argument("--clean-spp", type=int, default=512, help="samples for the target")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--oidn",
        action="store_true",
        help="render only the fast pass with Cycles' own denoiser on, as a baseline",
    )
    parser.add_argument(
        "--procedural",
        action="store_true",
        help="build a random scene instead of using the loaded .blend",
    )
    return parser.parse_args(argv)


def configure_cycles(scene, resolution, denoise=False):
    """Set the renderer up for physically linear, undenoised output."""
    scene.render.engine = "CYCLES"

    # Prefer the GPU. On Apple Silicon this is Metal; the try/except keeps the
    # script working on machines where no compute device is available.
    try:
        preferences = bpy.context.preferences.addons["cycles"].preferences
        for device_type in ("METAL", "CUDA", "OPTIX", "HIP"):
            try:
                preferences.compute_device_type = device_type
                break
            except TypeError:
                continue
        preferences.get_devices()
        for device in preferences.devices:
            device.use = True
        scene.cycles.device = "GPU"
    except Exception as error:  # noqa: BLE001 - informative, not fatal
        print(f"  (falling back to CPU rendering: {error})")
        scene.cycles.device = "CPU"

    # Decision 1: no denoising anywhere -- except when deliberately producing
    # the OpenImageDenoise baseline, which is the whole point of that mode.
    scene.cycles.use_denoising = denoise
    scene.cycles.use_adaptive_sampling = False  # fixed sample counts, so pairs match

    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False

    # Decision 4: linear data in, linear data out. "Raw" applies no tone curve,
    # which is what makes the saved EXR physically linear. The display device is
    # deliberately left alone: it only affects on-screen preview, and Blender 5
    # offers no "None" option for it.
    scene.view_settings.view_transform = "Raw"
    scene.view_settings.look = "None"


def enable_passes(view_layer):
    view_layer.use_pass_combined = True
    view_layer.use_pass_diffuse_color = True
    view_layer.use_pass_normal = True


def build_output_graph(scene):
    """Wire the compositor to write every pass into one multilayer EXR.

    Blender 5 splits these two capabilities: `scene.render.image_settings`
    offers OPEN_EXR but not the multilayer variant, while the File Output node
    offers only the multilayer variant. Auxiliary passes therefore have to go
    through the compositor, and they arrive as named layers inside a single
    file rather than as separate images.

    The API moved as well: the scene references a node group through
    `compositing_node_group` instead of owning `node_tree`, and the File Output
    node replaced `base_path`/`file_slots` with `directory`, `file_name` and
    `file_output_items`. The "DiffCol" socket is now "Diffuse Color".
    """
    tree = bpy.data.node_groups.new("Compositing", "CompositorNodeTree")
    scene.compositing_node_group = tree
    tree.nodes.clear()

    render_layers = tree.nodes.new("CompositorNodeRLayers")
    render_layers.scene = scene

    output = tree.nodes.new("CompositorNodeOutputFile")
    output.format.file_format = "OPEN_EXR_MULTILAYER"
    output.format.color_depth = "32"
    output.format.exr_codec = "ZIP"
    output.file_output_items.clear()

    slots = {"beauty": "Image"}
    slots.update(AUX_PASSES)
    for slot_name, socket_name in slots.items():
        if socket_name not in render_layers.outputs:
            # Loud, because a silently blank auxiliary buffer would poison the
            # dataset without ever raising an error.
            print(f"  WARNING: pass {socket_name!r} unavailable; {slot_name} would be blank")
            continue
        output.file_output_items.new("RGBA", slot_name)
        tree.links.new(render_layers.outputs[socket_name], output.inputs[slot_name])

    return output


def scene_bounds():
    """Centre and radius of everything visible, used to frame the camera."""
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not meshes:
        return Vector((0, 0, 0)), 5.0

    corners = [
        obj.matrix_world @ Vector(corner)
        for obj in meshes
        for corner in obj.bound_box
    ]
    minimum = Vector((min(c[i] for c in corners) for i in range(3)))
    maximum = Vector((max(c[i] for c in corners) for i in range(3)))
    centre = (minimum + maximum) / 2
    radius = max((maximum - minimum).length / 2, 1e-3)
    return centre, radius


def place_camera(scene, rng):
    """Put the camera at a random point on a sphere around the scene."""
    camera = scene.camera
    if camera is None:
        camera_data = bpy.data.cameras.new("Camera")
        camera = bpy.data.objects.new("Camera", camera_data)
        scene.collection.objects.link(camera)
        scene.camera = camera

    centre, radius = scene_bounds()
    distance = radius * rng.uniform(2.2, 3.4)
    azimuth = rng.uniform(0, 2 * math.pi)
    # Bias towards eye level rather than looking straight down.
    elevation = rng.uniform(math.radians(5), math.radians(55))

    camera.location = centre + Vector(
        (
            distance * math.cos(elevation) * math.cos(azimuth),
            distance * math.cos(elevation) * math.sin(azimuth),
            distance * math.sin(elevation),
        )
    )
    direction = centre - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def build_procedural_scene(rng):
    """A random still life: varied shapes, materials and lighting.

    Not a substitute for real production scenes, but it gives the network a
    spread of materials and light transport to learn from without needing any
    assets downloaded first.
    """
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene

    # Ground plane.
    bpy.ops.mesh.primitive_plane_add(size=40, location=(0, 0, 0))
    floor = bpy.context.active_object
    floor.data.materials.append(random_material(rng, "floor"))

    for index in range(rng.randint(4, 9)):
        spawn = rng.choice(
            [
                bpy.ops.mesh.primitive_uv_sphere_add,
                bpy.ops.mesh.primitive_cube_add,
                bpy.ops.mesh.primitive_torus_add,
                bpy.ops.mesh.primitive_monkey_add,
                bpy.ops.mesh.primitive_cone_add,
            ]
        )
        spawn(location=(rng.uniform(-4, 4), rng.uniform(-4, 4), rng.uniform(0.5, 3)))
        obj = bpy.context.active_object
        obj.rotation_euler = [rng.uniform(0, math.pi * 2) for _ in range(3)]
        obj.data.materials.append(random_material(rng, f"mat{index}"))

    # A couple of area lights, which produce soft shadows and therefore the
    # kind of low-frequency noise a denoiser has to handle.
    for _ in range(rng.randint(1, 3)):
        bpy.ops.object.light_add(
            type="AREA",
            location=(rng.uniform(-8, 8), rng.uniform(-8, 8), rng.uniform(5, 11)),
        )
        light = bpy.context.active_object
        light.data.energy = rng.uniform(300, 1600)
        light.data.size = rng.uniform(0.5, 5.0)

    world = bpy.data.worlds.new("World")
    scene.world = world
    ensure_nodes(world)
    world.node_tree.nodes["Background"].inputs[1].default_value = rng.uniform(0.02, 0.5)
    return scene


def random_material(rng, name):
    """A Principled BSDF spanning diffuse, glossy, metal and glass."""
    material = bpy.data.materials.new(name)
    ensure_nodes(material)
    bsdf = material.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (
        rng.random(),
        rng.random(),
        rng.random(),
        1.0,
    )
    bsdf.inputs["Roughness"].default_value = rng.random() ** 2
    bsdf.inputs["Metallic"].default_value = 1.0 if rng.random() < 0.25 else 0.0
    if rng.random() < 0.12 and "Transmission Weight" in bsdf.inputs:
        bsdf.inputs["Transmission Weight"].default_value = 1.0
    return material


def render_view(scene, output_node, directory, name, samples):
    """Render the current camera at a fixed sample count.

    Blender 5's File Output node writes only multilayer EXR, so the compositor
    buys nothing here: setting the render output format directly produces the
    same single file containing every enabled pass as a named layer, with less
    machinery to break.
    """
    scene.cycles.samples = samples
    # Decision 3: identical seed for both halves of the pair.
    scene.cycles.seed = 0
    scene.cycles.use_animated_seed = False

    output_node.directory = directory
    output_node.file_name = name
    bpy.ops.render.render(write_still=False)


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    os.makedirs(args.out, exist_ok=True)

    for view_index in range(args.views):
        if args.procedural:
            scene = build_procedural_scene(rng)
        else:
            scene = bpy.context.scene

        configure_cycles(scene, args.resolution, denoise=args.oidn)
        enable_passes(scene.view_layers[0])
        scene.frame_set(1)
        place_camera(scene, rng)

        view_dir = os.path.join(args.out, f"view_{view_index:04d}")
        os.makedirs(view_dir, exist_ok=True)
        output_node = build_output_graph(scene)

        if args.oidn:
            # Same scene, same camera, same sample count as the training input;
            # the only difference is that Cycles' denoiser is switched on. That
            # makes it a fair comparison against the learned model.
            print(f"[{view_index + 1}/{args.views}] oidn ({args.noisy_spp} spp, denoised)")
            render_view(scene, output_node, view_dir, "oidn", args.noisy_spp)
            continue

        # The noisy render also supplies the auxiliary buffers (decision 2).
        print(f"[{view_index + 1}/{args.views}] noisy ({args.noisy_spp} spp)")
        render_view(scene, output_node, view_dir, "noisy", args.noisy_spp)

        print(f"[{view_index + 1}/{args.views}] clean ({args.clean_spp} spp)")
        render_view(scene, output_node, view_dir, "clean", args.clean_spp)

    print(f"\nDone. {args.views} views written to {args.out}")


if __name__ == "__main__":
    main()
