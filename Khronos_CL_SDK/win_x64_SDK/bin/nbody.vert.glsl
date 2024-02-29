#version 330

// VS locations
#define POSITION    0
#define COLOR       1

// FS locations
#define FRAG_COLOR  0

layout(location = POSITION) in vec3 in_Position;
layout(location = COLOR) in float in_Color;

out block
{
    float Color;
} VS_Out;

uniform mat4 mat_MVP;

void main()
{
    gl_Position = mat_MVP * vec4(in_Position, 1.0);

    VS_Out.Color = in_Color;
}
