#version 330

// VS locations
#define POSITION    0
#define COLOR       1

// FS locations
#define FRAG_COLOR  0

in block
{
    float Color;
} FS_In;

out vec4 FragColor;

void main()
{
    float factor = FS_In.Color / 500.f;
    FragColor = vec4(factor, factor, factor ,1.0f);
}
