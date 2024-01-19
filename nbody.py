import sys, argparse

import taichi as ti

# python3 nbody.py --arch=cpu --body=8 --fps=-1
# python3 nbody.py --arch=vulkan --body=8 --fps=60

# -----------------------------------------------------------------------------------------------------------

@ti.data_oriented
class NBodySystem:

    def __init__(self, nb_body=8):

        self.nb_body = nb_body

        self.pos = ti.Vector.field(3, dtype=ti.f32, shape=self.nb_body)
        self.vel = ti.Vector.field(3, dtype=ti.f32, shape=self.nb_body)
        self.acc = ti.Vector.field(3, dtype=ti.f32, shape=self.nb_body)

        self.mass = 1.0

    @ti.kernel
    def init(self):
        for i in range(self.nb_body):
            self.pos[i] = [((ti.random(float) * 2) - 1) * 1.0, ((ti.random(float) * 2) - 1) * 1.0, ((ti.random(float) * 2) - 1) * 1.0]
            self.vel[i] = [0.0, 0.0, 0.0]
            self.acc[i] = [0.0, 0.0, 0.0]

    @ti.kernel
    def update(self, dt: ti.f32, eps: ti.f32):

        for i in range(self.nb_body):
            self.vel[i] += self.acc[i] * 0.5 * dt
            self.pos[i] += self.vel[i] * dt

        # only the outer loop is optimized 
            
        # For tne next for loop set the number of threads in a block on GPU
        # ti.loop_config(block_dim=8)

        # For tne next for loop set the number of threads in a block on CPU
        #ti.loop_config(parallelize=8)

        #for i, j in ti.ndrange(self.nb_body, self.nb_body): # does not work on vulkan / opengl ?
        for i in range(self.nb_body):
            for j in range(self.nb_body):
                if i != j:
                    DR = self.pos[j] - self.pos[i]
                    DR2 = ti.math.dot(DR, DR)
                    DR2 += eps*eps

                    PHI = self.mass / (ti.sqrt(DR2) * DR2)

                    self.acc[i] += DR * PHI

        for i in range(self.nb_body):
            self.vel[i] += self.acc[i] * 0.5 * dt
            self.acc[i] = [0.0, 0.0, 0.0]  
          
# -----------------------------------------------------------------------------------------------------------

class App:

    def __init__(self, screen_width=1280, screen_height=800, max_fps=-1, camera_pos=ti.Vector([0.0, 0.0, 8.0]), nb_body=8, dt=0.005, eps=0.5):

        # Body
        self.nb_body = nb_body
        self.dt = dt
        self.eps = eps

        self.nbody_system = NBodySystem(nb_body=self.nb_body)
        self.nbody_system.init()

        # Window
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.camera_pos = camera_pos
        self.max_fps = max_fps

        self.window = ti.ui.Window("LeapFrog N-body", res=(self.screen_width, self.screen_height), fps_limit=self.max_fps, vsync=0)

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
        self.dt = self.window.GUI.slider_float("dt", self.dt, minimum=0.0, maximum=0.1)
        self.eps = self.window.GUI.slider_float("eps", self.eps, minimum=0.01, maximum=1.0)
        self.window.GUI.end()

    def run(self):

        # drawing loop
        while self.window.running:

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

            self.nbody_system.update(self.dt, self.eps)
            #ti.profiler.print_kernel_profiler_info()

            self.scene.ambient_light((0.8, 0.8, 0.8))
            self.scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))
            
            self.scene.particles(self.nbody_system.pos, radius=0.01, color=(1.0, 1.0, 1.0))

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
    parser = argparse.ArgumentParser(description="Leapfrog N-Body")

    parser.add_argument('-a', '--arch', help='Taichi backend', default="cpu", action="store")
    parser.add_argument('-f', '--fps', help='Max FPS, -1 for unlimited', default=-1, type=int)
    parser.add_argument('-b', '--body', help='NB Body', default=64, type=int)

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
    app = App(screen_width=SCREEN_WIDTH, screen_height=SCREEN_HEIGHT, max_fps=args["fps"], camera_pos=CAMERA_POS, 
              nb_body=args["body"], dt=0.005, eps=0.5)
    app.run()

if __name__ == "__main__":
    main()