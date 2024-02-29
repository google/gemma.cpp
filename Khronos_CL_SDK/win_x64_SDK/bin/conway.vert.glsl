#version 330

// VS locations
#define POSITION    0
#define TEXCOORD    1

layout(location = POSITION) in vec2 in_Position;
layout(location = TEXCOORD) in vec2 in_TexCoord;

out block
{
    vec2 TexCoord;
} VS_Out;

void main()
{
    gl_Position = vec4(in_Position, 0.0, 1.0);

    VS_Out.TexCoord = in_TexCoord;
}
