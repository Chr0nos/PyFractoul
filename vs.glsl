#version 400 core

uniform mat4 mv_matrix;
uniform mat4 projection;

attribute vec3 vertex;
attribute vec2 uv;

out vec2 UV;

void main() {
  // gl_Position = projection * mv_matrix * vec4(vertex, 1.0);
  gl_Position = vec4(vertex, 1.0);
  UV = uv;
}
