import glfw
import numpy as np
from pyrr import Matrix44
from OpenGL.GL import glGenVertexArrays, glDeleteBuffers, glBufferData, \
    GL_ARRAY_BUFFER, glBindVertexArray, glBindBuffer, glGenBuffers, GL_FLOAT, \
    GL_FALSE, glVertexAttribPointer, shaders, GL_STATIC_DRAW, \
    glGetAttribLocation, glEnableVertexAttribArray, glUniformMatrix4fv


# https://rdmilligan.wordpress.com/2016/09/03/opengl-vao-using-python/


class GlfwError(Exception):
    pass


class Shader:
    def __init__(self, code: str, stype):
        self.id = shaders.compileShader(code, stype)

    def __del__(self):
        if hasattr(self, 'id'):
            shaders.glDeleteShader(self.id)

    @classmethod
    def load(cls, filepath, stype):
        with open(filepath, 'r') as fp:
            return cls(fp.read(), stype)


class Program:
    def __init__(self, vs: Shader, fs: Shader):
        self.vs = vs
        self.fs = fs
        self.id = shaders.compileProgram(vs.id, fs.id, validate=False)
        shaders.glUseProgram(self.id)

    def __del__(self):
        del self.vs
        del self.fs

    def uniform(self, name):
        return shaders.glGetUniformLocation(self.id, name)

    def attribute(self, name):
        return shaders.glGetAttribLocation(self.id, name)

    def send_attr(self, name, vertex, total_size,
                  step=3, dtype=GL_FLOAT, normalize=GL_FALSE) -> int:
        position = shaders.glGetAttribLocation(self.id, name)
        if position < 0:
            raise ValueError(f'{position} {name}')
        glVertexAttribPointer(position, step, dtype, normalize, total_size, vertex)
        glEnableVertexAttribArray(position)
        return position

    def send_matrix(self, name, matrix, func=glUniformMatrix4fv):
        position = shaders.glGetUniformLocation(self.id, name)
        if position < 0:
            raise ValueError(f'{position} {name} in {self.id}')
        return func(position, 1, GL_FALSE, matrix)


class Buffer:
    def __init__(self, kind=GL_ARRAY_BUFFER):
        self.id = glGenBuffers(1)
        self.kind = kind

    def send_vertices(self, vertices):
        vertices = np.array(vertices, np.float32)
        glBindBuffer(GL_ARRAY_BUFFER, self.id)
        glBufferData(self.kind, len(vertices) * 4, vertices, GL_STATIC_DRAW)
        # glBindBuffer(GL_ARRAY_BUFFER, self.id)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        return self

    def __del__(self):
        glDeleteBuffers(1, [self.id])


class Vao:
    def __init__(self, vertices: np.ndarray):
        self.vbo = Buffer().send_vertices(vertices)
        self.id = glGenVertexArrays(1)
        glBindVertexArray(self.id)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo.id)

    def __del__(self):
        glDeleteBuffers(1, [self.id])


class GlWindow:
    def __init__(self, w, h, title: str):
        self.window = glfw.create_window(w, h, title, None, None)
        self.height = h
        self.width = w
        if not self.window:
            raise GlfwError

    def run_loop(self, *args, **kwargs) -> None:
        glfw.make_context_current(self.window)
        while not glfw.window_should_close(self.window):
            glfw.poll_events()
            self.render()
            glfw.swap_buffers(self.window)

    def render(self, *args, **kwargs) -> None:
        pass

    @property
    def ratio(self):
        return self.width / self.height


def vec_iter(vec_list, vsize=3):
    i = 0
    m = len(vec_list)
    while i < m:
        point = vec_list[i:i + vsize]
        yield point
        i += vsize


def main():
    if not glfw.init():
        return
    print(glfw.get_version_string())
    window = GlWindow(640, 480, 'Hello world')
    glfw.make_context_current(window.window)

    vao = Vao((10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0))
    program = Program(
        Shader.load('vs.glsl', shaders.GL_VERTEX_SHADER),
        Shader.load('fs.glsl', shaders.GL_FRAGMENT_SHADER)
    )

    # two tringles making a square.
    plane = (
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,

        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        1.0, 1.0, 0.0
    )
    matrix_projection = Matrix44.perspective_projection(75, window.ratio, 1, 1000)
    matrix_move = Matrix44.identity()
    print(matrix_projection)
    print(matrix_move)

    program.send_attr('vertex', plane, step=3, total_size=len(plane) * 3)
    program.send_matrix('projection', matrix_projection)
    window.run_loop()
    glfw.terminate()


if __name__ == "__main__":
    main()
