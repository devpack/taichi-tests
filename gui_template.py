import sys, argparse

import taichi as ti

# -----------------------------------------------------------------------------------------------------------

class App:

    def __init__(self, screen_width=1280, screen_height=800, max_fps=-1, camera_pos=ti.Vector([0.0, 0.0, 8.0])):

        # Window
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.camera_pos = camera_pos
        self.max_fps = max_fps

        self.window = ti.ui.Window("", res=(self.screen_width, self.screen_height), fps_limit=self.max_fps, vsync=0)

        self.canvas = self.window.get_canvas()
        self.scene = self.window.get_scene()

        self.camera = ti.ui.Camera()
        self.camera.position(self.camera_pos.x, self.camera_pos.y, self.camera_pos.z)
        self.camera.lookat(0.0, 0.0, 0.0)
        self.camera.up(0, 1, 0)
        self.camera.projection_mode(ti.ui.ProjectionMode.Perspective)
        self.scene.set_camera(self.camera)

        self.canvas.set_background_color((0.1, 0.1, 0.1))
        
        self.cam_moved = False

    def show_options(self):
        self.window.GUI.begin("Options", 0.05, 0.1, 0.2, 0.15)
        self.max_fps = self.window.GUI.slider_float("dt", self.max_fps, minimum=0, maximum=9999)
        self.window.GUI.end()

    def run(self):

        # drawing loop
        while self.window.running:

            #self.window

            self.camera.track_user_inputs(self.window, movement_speed=1.0, hold_key=ti.ui.RMB)

            for e in self.window.get_events(ti.ui.PRESS):
                if e.key in [ti.ui.ESCAPE]:
                    exit()
                if e.key in [ti.ui.DOWN]:
                    self.camera_pos.z += 1.001
                    self.cam_moved = True
                if e.key in [ti.ui.UP]:
                    self.camera_pos.z -= 1.001
                    self.cam_moved = True

            if self.cam_moved:
                self.camera.position(self.camera_pos.x, self.camera_pos.y, self.camera_pos.z)
                self.camera.lookat(0.0, 0.0, 0.0)
                self.cam_moved = False

            self.scene.set_camera(self.camera)
            self.scene.ambient_light((0.8, 0.8, 0.8))
            self.scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))
            
            #pos = ti.Vector.field(3, dtype=float, shape=1)
            #pos[0] = [0.0, 0.0, 0.0]
            #self.scene.particles(pos, radius=0.01, color=(1.0, 1.0, 1.0))

            self.canvas.scene(self.scene)
                        
            self.show_options()
            self.window.show()

        self.window.destroy()

# -----------------------------------------------------------------------------------------------------------

def main():

    # const
    USE_PROFILER  = 0

    SCREEN_WIDTH  = 1280
    SCREEN_HEIGHT = 800
    CAMERA_POS = ti.Vector([0.0, 0.0, 8.0])

    # args
    parser = argparse.ArgumentParser(description="")

    parser.add_argument('-a', '--arch', help='Taichi backend', default="cpu", action="store")
    parser.add_argument('-f', '--fps', help='Max FPS, -1 for unlimited', default=-1, type=int)

    result = parser.parse_args()
    args = dict(result._get_kwargs())

    print("Args = %s" % args)

    if args["arch"] in ("cpu", "x64"):
        ti.init(ti.cpu, debug=0, default_ip=ti.i32, default_fp=ti.f32, kernel_profiler=USE_PROFILER)

    elif args["arch"] in ("gpu", "cuda"):
        ti.init(ti.gpu, kernel_profiler=USE_PROFILER)
    elif args["arch"] in ("opengl",):
        ti.init(ti.opengl)
    elif args["arch"] in ("vulkan",):
        ti.init(ti.vulkan)

    # App
    app = App(screen_width=SCREEN_WIDTH, screen_height=SCREEN_HEIGHT, max_fps=args["fps"], camera_pos=CAMERA_POS)
    app.run()

if __name__ == "__main__":
    main()