#version 330

// VS locations
#define POSITION    0
#define TEXCOORD    1

in block
{
    vec2 TexCoord;
} FS_In;

out vec4 FragColor;

uniform usampler2D texsampler;

void main()
{
    FragColor = texture(texsampler, FS_In.TexCoord);
}
